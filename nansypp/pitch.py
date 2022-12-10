import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block, 
    """
    def __init__(self, in_channels: int, out_channels: int, kernels: int):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output channels.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, (kernels, 1), padding=(kernels // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, (kernels, 1), padding=(kernels // 2, 0)))

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, in_channels, F, N]], input channels.
        Returns:
            [torch.float32; [B, out_channels, F // 2, N]], output channels.
        """
        # [B, out_channels, F, N]
        outputs = self.branch(inputs)
        # [B, out_channels, F, N]
        shortcut = self.shortcut(inputs)
        # [B, out_channels, F // 2, N]
        return F.avg_pool2d(outputs + shortcut, (2, 1))


def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Exponential sigmoid.
    Args:
        x: [torch.float32; [...]], input tensors.
    Returns:
        sigmoid outputs.
    """
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7


class PitchEncoder(nn.Module):
    """Pitch-encoder.
    """
    def __init__(self,
                 freq: int,
                 prekernels: int,
                 kernels: int,
                 channels: int,
                 blocks: int,
                 gru: int,
                 hiddens: int,
                 f0_bins: int):
        """Initializer.
        Args:
            freq: the number of the frequency bins.
            prekernels: size of the first convolutional kernels.
            kernels: size of the frequency-convolution kernels.
            channels: size of the channels.
            blocks: the number of the residual blocks.
            gru: size of the GRU hidden states.
            hiddens: size of the hidden channels.
            f0_bins: size of the output f0-bins.
        """
        super().__init__()
        self.f0_bins = f0_bins
        # prekernels=7
        self.preconv = nn.Conv1d(
            1, channels, prekernels, padding=prekernels // 2)
        # channels=128, kernels=3, blocks=2
        self.resblock = nn.Sequential(*[
            ResBlock(channels, channels, kernels)
            for _ in range(blocks)])
        # unknown `gru`
        self.gru = nn.GRU(
            freq * channels // (2 * blocks), gru,
            batch_first=True, bidirectional=True)
        # unknown `hiddens`
        # f0_bins=64
        self.proj = nn.Sequential(
            nn.Linear(gru * 2, hiddens * 2),
            nn.ReLU(),
            nn.Linear(hiddens * 2, f0_bins + 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the pitch from inputs.
        Args:
            inputs: [torch.float32; [B, F, N]], input tensor.
        Returns:
            f0: [torch.float32; [B, N, f0_bins]], f0 outputs, based on frequency bins.
            p_amp, ap_amp: [torch.float32; [B, N]], amplitude values.
        """
        # B, _, N
        bsize, _, timesteps = inputs.shape
        # [B, C, F, N]
        x = self.preconv(inputs[:, None])
        # [B, C F // 4, N]
        x = self.resblock(x)
        # [B, N, C x F // 4]
        x = x.transpose(0, 3, 1, 2).reshape(bsize, timesteps, -1)
        # [B, N, G x 2]
        x, _ = self.gru(x)
        # [B, N, f0_bins], [B, N, 1], [B, N, 1]
        f0, p_amp, ap_amp = torch.split(self.proj(x), [self.f0_bins, 1, 1], dim=-1)
        return \
            torch.softmax(f0, dim=-1), \
            exponential_sigmoid(p_amp).squeeze(dim=-1), \
            exponential_sigmoid(ap_amp).squeeze(dim=-1)
