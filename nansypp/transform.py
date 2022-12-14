from typing import Optional

import torch
import torch.nn as nn
import torchaudio

from .rough_cqt import CQT2010v2


class ConstantQTransform(nn.Module):
    """Constant Q-Transform.
    """
    def __init__(self,
                 strides: int,
                 fmin: float,
                 fmax: float,
                 bins: int,
                 bins_per_octave: int,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: the number of the samples between adjacent frame.
            fmin, fmax: frequency min, max.
            bins: the number of the output bins.
            bins_per_octave: the number of the frequency bins per octave.
            sr: sampling rate.
        """
        super().__init__()
        # unknown `strides`
        # , since linguistic information is 50fps, strides could be 441
        # fmin=32.7(C0), fmax=8000
        # bins=191, bins_per_octave=24
        self.cqt = CQT2010v2(
            sr,
            strides,
            fmin,
            fmax,
            n_bins=bins,
            bins_per_octave=bins_per_octave,
            trainable=False,
            output_format='Magnitude')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CQT on inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, bins, T / strides]], CQT magnitudes.
        """
        return self.cqt(inputs[:, None])


class MelSpectrogram(nn.Module):
    """log-Mel scale spectrogram.
    """
    def __init__(self,
                 strides: int,
                 windows: int,
                 mel: int,
                 fmin: int = 0,
                 fmax: Optional[int] = 8000,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: hop length, the number of the frames between adjacent windows.
            windows: length of the windows.
            mel: size of the mel filterbanks.
            fmin, fmax: minimum, maximum frequency,
                if fmax is None, use half of the sample rate as default.
            sr: sample rate.
        """
        super().__init__()
        self.strides, self.windows = strides, windows
        # [mel, windows // 2 + 1]
        # use slaney-scale mel filterbank for `librosa.filters.mel` compatibility.
        self.register_buffer(
            'melfilter',
            torchaudio.functional.melscale_fbanks(
                windows // 2 + 1,
                fmin, fmax, mel, sr,
                norm='slaney', mel_scale='slaney').T,
            persistent=False)
        # [windows], use hann window
        self.register_buffer(
            'hann', torch.hann_window(windows), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate the log-mel scale spectrogram.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, mel, T / strides]], log-mel spectrogram
        """
        # [B, windows // 2 + 1, T / strides, 2]
        fft = torch.stft(
            audio, self.windows, self.strides,
            window=self.hann,
            center=True, pad_mode='reflect', return_complex=False)
        # [B, windows // 2 + 1, T / strides]
        mag = torch.sqrt(fft.square().sum(dim=-1) + 1e-7)
        # [B, mel, T / strides]
        return torch.log(torch.matmul(self.melfilter, mag) + 1e-7)
