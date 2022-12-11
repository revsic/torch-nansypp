from typing import List

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
    def __init__(self, in_channels: int, hiddens: int, heads: int):
        """
        """


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
                 hiddens: int):
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
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(len(dilations) * channels, hiddens, 1),
            nn.ReLU())
        # attentive pooling and additional projector
        # out_channels=192
        self.pool = nn.Sequential(
            AttentiveStatisticsPooling(hiddens, bottleneck),
            nn.BatchNorm1d(hiddens * 2),
            nn.Linear(hiddens * 2, out_channels),
            nn.BatchNorm1d(out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the x-vectors from the input sequence.
        Args:
            inputs: [torch.float32; [B, I, T]], input sequences,
                where I = `in_channels`.
        Returns:
            [torch.float32; [B, O]], x-vectors,
                where O = `out_channels`.
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
        x = self.conv1x1(torch.cat(xs, dim=1))
        # [B, O]
        return F.normalize(self.pool(x), p=2, dim=-1)