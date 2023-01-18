from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Block(nn.Module):
    """Multi-scale residual blocks.
    """
    def __init__(self, channels: int, scale: int, kernels: int, dilation: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the blocks.
            kenels: size of the convolutional kernels.
            dilation: dilation factors.
        """
        super().__init__()
        assert channels % scale == 0, \
            f'size of the input channels(={channels})' \
            f' should be factorized by scale(={scale})'
        width = channels // scale
        self.scale = scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    width, width, kernels,
                    padding=(kernels - 1) * dilation // 2,
                    dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(width))
            for _ in range(scale - 1)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D tensor,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, W, T], (S - 1) x [B, W, T] where W = C // S
        straight, *xs = inputs.chunk(self.scale, dim=1)
        # [B, W, T]
        base = torch.zeros_like(straight)
        # S x [B, W, T]
        outs = [straight]
        for x, conv in zip(xs, self.convs):
            # [B, W, T], increasing receptive field progressively
            base = conv(x + base)
            outs.append(base)
        # [B, C, T]
        return torch.cat(outs, dim=1)


class SERes2Block(nn.Module):
    """Multiscale residual block with Squeeze-Excitation modules.
    """
    def __init__(self,
                 channels: int,
                 scale: int,
                 kernels: int,
                 dilation: int,
                 bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the resolutions, for res2block.
            kernels: size of the convolutional kernels.
            dilation: dilation factor.
            bottleneck: size of the bottleneck layers for squeeze and excitation.
        """
        super().__init__()
        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        
        self.res2block = Res2Block(channels, scale, kernels, dilation)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, C, T]
        x = self.preblock(inputs)
        # [B, C, T], res2net, multi-scale architecture
        x = self.res2block(x)
        # [B, C, T]
        x = self.postblock(x)
        # [B, C], squeeze and excitation
        scale = self.excitation(x.mean(dim=-1))
        # [B, C, T]
        x = x * scale[..., None]
        # residual connection
        return x + inputs


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling.
    """
    def __init__(self, channels: int, bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            bottleneck: size of the bottleneck.
        """
        super().__init__()
        # nonlinear=Tanh
        # ref: https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding
        # ref: https://github.com/TaoRuijie/ECAPA-TDNN
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Softmax(dim=-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pooling with weighted statistics.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C x 2]], weighted statistics.
        """
        # [B, C, T]
        weights = self.attention(inputs)
        # [B, C]
        mean = torch.sum(weights * inputs, dim=-1)
        var = torch.sum(weights * inputs ** 2, dim=-1) - mean ** 2
        # [B, C x 2], for numerical stability of square root
        return torch.cat([mean, (var + 1e-7).sqrt()], dim=-1)


class MultiheadAttention(nn.Module):
    """Multi-head scaled dot-product attention.
    """
    def __init__(self,
                 keys: int,
                 values: int,
                 queries: int,
                 out_channels: int,
                 hiddens: int,
                 heads: int):
        """Initializer.
        Args:
            keys, valeus, queries: size of the input channels.
            out_channels: size of the output channels.
            hiddens: size of the hidden channels.
            heads: the number of the attention heads.
        """
        super().__init__()
        assert hiddens % heads == 0, \
            f'size of hiddens channels(={hiddens}) should be factorized by heads(={heads})'
        self.channels, self.heads = hiddens // heads, heads
        self.proj_key = nn.Conv1d(keys, hiddens, 1)
        self.proj_value = nn.Conv1d(values, hiddens, 1)
        self.proj_query = nn.Conv1d(queries, hiddens, 1)
        self.proj_out = nn.Conv1d(hiddens, out_channels, 1)

    def forward(self,
                keys: torch.Tensor,
                values: torch.Tensor,
                queries: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform the inputs.
        Args:
            keys: [torch.float32; [B, keys, S]], attention key.
            values: [torch.float32; [B, values, S]], attention value.
            queries: [torch.float32; [B, queries, T]], attention query.
            mask: [torch.float32; [B, S, T]], attention mask, 0 for paddings.
        Returns:
            [torch.float32; [B, out_channels, T]], transformed outputs.
        """
        # B, T
        bsize, _, querylen = queries.shape
        # S
        keylen = keys.shape[-1]
        assert keylen == values.shape[-1], 'lengths of key and value are not matched'
        # [B, H, hiddens // H, S]
        keys = self.proj_key(keys).view(bsize, self.heads, -1, keylen)
        values = self.proj_value(values).view(bsize, self.heads, -1, keylen)
        # [B, H, hiddens // H, T]
        queries = self.proj_query(queries).view(bsize, self.heads, -1, querylen)
        # [B, H, S, T]
        score = torch.matmul(keys.transpose(2, 3), queries) * (self.channels ** -0.5)
        if mask is not None:
            score.masked_fill_(~mask[:, None, :, :1].to(torch.bool), -np.inf)
        # [B, H, S, T]
        weights = torch.softmax(score, dim=2)
        # [B, out_channels, T]
        out = self.proj_out(
            torch.matmul(values, weights).view(bsize, -1, querylen))
        if mask is not None:
            out = out * mask[:, :1]
        return out


class TimberEncoder(nn.Module):
    """ECAPA-TDNN: Emphasized Channel Attention,
    [1] Propagation and Aggregation in TDNN Based Speaker Verification,
        Desplanques et al., 2020, arXiv:2005.07143.
    [2] Res2Net: A New Multi-scale Backbone architecture,
        Gao et al., 2019, arXiv:1904.01169.
    [3] Squeeze-and-Excitation Networks, Hu et al., 2017, arXiv:1709.01507.
    [4] Attentive Statistics Pooling for Deep Speaker Embedding,
        Okabe et al., 2018, arXiv:1803.10963.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 prekernels: int,
                 scale: int,
                 kernels: int,
                 dilations: List[int],
                 bottleneck: int,
                 hiddens: int,
                 latent: int,
                 timber: int,
                 tokens: int,
                 heads: int,
                 contents: int,
                 slerp: float):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output embeddings.
            channels: size of the major states.
            prekernels: size of the convolutional kernels before feed to SERes2Block.
            scale: the number of the resolutions, for SERes2Block.
            kernels: size of the convolutional kernels, for SERes2Block.
            dilations: dilation factors.
            bottleneck: size of the bottleneck layers,
                both SERes2Block and AttentiveStatisticsPooling.
            hiddens: size of the hidden channels for attentive statistics pooling.
            latent: size of the timber latent query.
            timber: size of the timber tokens.
            tokens: the number of the timber tokens.
            heads: the number of the attention heads, for timber token block.
            contents: size of the content queries.
            slerp: weight value for spherical interpolation.
        """
        super().__init__()
        # channels=512, prekernels=5
        # ref:[1], Figure2 and Page3, "architecture with either 512 or 1024 channels"
        self.preblock = nn.Sequential(
            nn.Conv1d(in_channels, channels, prekernels, padding=prekernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        # scale=8, kernels=3, dilations=[2, 3, 4], bottleneck=128
        self.blocks = nn.ModuleList([
            SERes2Block(channels, scale, kernels, dilation, bottleneck)
            for dilation in dilations])
        # hiddens=1536
        # TODO: hiddens=3072 on NANSY++
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(len(dilations) * channels, hiddens, 1),
            nn.ReLU())
        # multi-head attention for time-varying timber 
        # NANSY++, latent=512, tokens=50
        self.timber_query = nn.Parameter(torch.randn(1, latent, tokens))
        # NANSY++, timber=128
        # unknown `heads`
        self.pre_mha = MultiheadAttention(
            hiddens, hiddens, latent, latent, latent, heads)
        self.post_mha = MultiheadAttention(
            hiddens, hiddens, latent, timber, latent, heads)
        # attentive pooling and additional projector
        # out_channels=192
        self.pool = nn.Sequential(
            AttentiveStatisticsPooling(hiddens, bottleneck),
            nn.BatchNorm1d(hiddens * 2),
            nn.Linear(hiddens * 2, out_channels),
            nn.BatchNorm1d(out_channels))

        # time-varying timber encoder
        self.timber_key = nn.Parameter(torch.randn(1, timber, tokens))
        self.sampler = MultiheadAttention(
            timber, timber, contents, timber, latent, heads)
        self.proj = nn.Conv1d(timber, out_channels, 1)
        # unknown `slerp`
        assert 0 <= slerp <= 1, f'value slerp(={slerp:.2f}) should be in range [0, 1]'
        self.slerp = slerp

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the x-vectors from the input sequence.
        Args:
            inputs: [torch.float32; [B, in_channels, T]], input sequences,
        Returns:
            [torch.float32; [B, out_channels]], global x-vectors,
            [torch.float32; [B, timber, tokens]], timber token bank.
        """
        # [B, C, T]
        x = self.preblock(inputs)
        # N x [B, C, T]
        xs = []
        for block in self.blocks:
            # [B, C, T]
            x = block(x)
            xs.append(x)
        # [B, H, T]
        mfa = self.conv1x1(torch.cat(xs, dim=1))
        # [B, O]
        global_ = F.normalize(self.pool(mfa), p=2, dim=-1)
        # B
        bsize, _ = global_.shape
        # [B, latent, tokens]
        query = self.timber_query.repeat(bsize, 1, 1)
        # [B, latent, tokens]
        query = self.pre_mha.forward(mfa, mfa, query) + query
        # [B, timber, tokens]
        local = self.post_mha.forward(mfa, mfa, query)
        # [B, out_channels], [B, timber, tokens]
        return global_, local

    def sample_timber(self,
                      contents: torch.Tensor,
                      global_: torch.Tensor,
                      tokens: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
        """Sample the timber tokens w.r.t. the contents.
        Args:
            contents: [torch.float32; [B, contents, T]], content queries.
            global_: [torch.float32; [B, out_channels]], global x-vectors, L2-normalized.
            tokens: [torch.float32; [B, timber, tokens]], timber token bank.
            eps: small value for preventing train instability of arccos in slerp.
        Returns:
            [torch.float32; [B, out_channels, T]], time-varying timber embeddings.
        """
        # [B, timber, tokens]
        key = self.timber_key.repeat(contents.shape[0], 1, 1)
        # [B, timber, T]
        sampled = self.sampler.forward(key, tokens, contents)
        # [B, out_channels, T]
        sampled = F.normalize(self.proj(sampled), p=2, dim=1)
        # [B, 1, T]
        theta = torch.matmul(global_[:, None], sampled).clamp(-1 +  eps, 1 - eps).acos()
        # [B, 1, T], slerp
        # clamp the theta is not necessary since cos(theta) is already clampped
        return (
            torch.sin(self.slerp * theta) * sampled
            + torch.sin((1 - self.slerp) * theta) * global_[..., None]) / theta.sin()
