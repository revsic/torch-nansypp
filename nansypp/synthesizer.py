from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .wavenet import WaveNet


class SignalGenerator(nn.Module):
    """Additive sinusoidal, subtractive filtered noise signal generator.
    """
    def __init__(self, scale: int, sr: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sr: sampling rate.
        """
        super().__init__()
        self.sr = sr
        self.upsampler = nn.Upsample(scale_factor=scale, mode='linear')

    def forward(self,
                pitch: torch.Tensor,
                p_amp: torch.Tensor,
                ap_amp: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp, ap_amp: [torch.float32; [B, N]], periodical, aperiodical amplitude.
            noise: [torch.float32; [B, T]], predefined noise, if provided.
        Returns:
            [torch.float32; [B, T(=N x scale)]], base signal.
        """
        # [B, T]
        pitch = self.upsampler(pitch[:, None]).squeeze(dim=1)
        p_amp = self.upsampler(p_amp[:, None]).squeeze(dim=1)
        # [B, T]
        x = p_amp * torch.cumsum(2 * np.pi * pitch / self.sr, dim=-1)
        # [B, T]
        ap_amp = self.upsampler(ap_amp[:, None]).squeeze(dim=1)
        if noise is None:
            # [B, T], U[-1, 1] sampled
            noise = torch.rand_like(x) * 2. - 1.
        # [B, T]
        y = ap_amp * noise
        return x + y


class Synthesizer(nn.Module):
    """Signal-level synthesizer.
    """
    def __init__(self,
                 scale: int,
                 sr: int,
                 channels: int,
                 aux: int,
                 kernels: int,
                 dilation_rate: int,
                 layers: int,
                 cycles: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sr: sampling rate.
            channels: size of the hidden channels.
            aux: size of the auxiliary input channels.
            kernels: size of the convolutional kernels.
            dilation_rate: dilaion rate.
            layers: the number of the wavenet blocks in single cycle.
            cycles: the number of the cycles.
        """
        super().__init__()
        self.excitation = SignalGenerator(scale, sr)

        self.wavenet = WaveNet(
            channels,
            aux,
            kernels,
            dilation_rate,
            layers,
            cycles)

    def forward(self,
                pitch: torch.Tensor,
                p_amp: torch.Tensor,
                ap_amp: torch.Tensor,
                frame: torch.Tensor,
                noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp, ap_amp: [torch.float32; [B, N]], periodical, aperiodical amplitude.
            frame: [torch.float32; [B, aux, N']], frame-level feature map.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation signal and generated signal.
        """
        # [B, T]
        excitation = self.excitation.forward(pitch, p_amp, ap_amp, noise=noise)
        # [B, aux, T]
        interp = F.interpolate(frame, size=excitation.shape[-1], mode='linear')
        # [B, T]
        signal = self.wavenet.forward(excitation, interp)
        return excitation, signal
