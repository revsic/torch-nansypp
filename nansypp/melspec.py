from typing import Optional

import torch
import torch.nn as nn
import torchaudio


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
