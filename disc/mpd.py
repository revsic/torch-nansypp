from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    """Period-aware discrminator.
    """
    def __init__(self,
                 channels: List[int],
                 period: int,
                 kernels: int,
                 strides: int,
                 postkernels: int,
                 leak: float):
        """Initializer.
        Args:
            channels: list of the channel sizes.
            period: size of the unit period.
            kernels: size of the convolutional kernel.
            strides: stride of the convolution.
            postkernels: size of the postnet convolutional kernel.
            leak: negative slope of leaky relu.
        """
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(
                    inc, outc, (kernels, 1), (strides, 1), padding=(kernels // 2, 0))),
                nn.LeakyReLU(leak))
            for inc, outc in zip([1] + channels, channels)])
        
        lastc = channels[-1]
        self.convs.append(
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(
                    lastc, lastc, (kernels, 1), padding=(kernels // 2, 0))),
                nn.LeakyReLU(leak)))

        self.postconv = nn.utils.weight_norm(nn.Conv2d(
            lastc, 1, (postkernels, 1), padding=(postkernels // 2, 0)))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Discriminate the inputs in multiple periods.
        Args:
            x: [torch.float32; [B, T]], input audio signal.
        Returns:
            outputs: [torch.float32; [B, S]], logits.
            fmap: [torch.float32; [B, C, F, P]], list of feature maps.
        """
        # B, T
        bsize, timestep = inputs.shape
        if timestep % self.period != 0:
            # padding for foldability
            padsize = self.period - timestep % self.period
            # [B, T + R]
            inputs = F.pad(inputs[:, None], (0, padsize), 'reflect').squeeze(1)
            # T + R
            timestep = timestep + padsize
        # [B, 1, F(=T // P), P]
        x = inputs.view(bsize, 1, timestep // self.period, self.period)
        # period-aware discrminator
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        # [B, 1, S', P]
        x = self.postconv(x)
        fmap.append(x)
        # [B, S]
        return x.view(bsize, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """MPD: Multi-period discriminator.
    """
    def __init__(self,
                 channels: List[int],
                 periods: List[int],
                 kernels: int,
                 strides: int,
                 postkernels: int,
                 leak: float):
        """Initializer.
        Args:
            param: hifigan parameters.
        """
        super().__init__()
        self.disc = nn.ModuleList([
            PeriodDiscriminator(
                channels,
                period,
                kernels,
                strides,
                postkernels,
                leak)
            for period in periods])

    def forward(self, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Discriminate the samples from real or fake.
        Args:
            x: [B, T], audio sample.
        Returns:
            multiple discriminating results and feature maps.
        """
        results, fmaps = [], []
        for disc in self.disc:
            result, fmap = disc(inputs)
            results.append(result)
            fmaps.append(fmap)
        return results, fmaps
