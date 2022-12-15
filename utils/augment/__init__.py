from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lpc import LinearPredictiveCoding
from .peq import ParametricEqualizer

from config import Config


class Augment(nn.Module):
    """Waveform augmentation.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: Nansy configurations.
        """
        super().__init__()
        self.config = config
        self.coder = LinearPredictiveCoding(
            config.train.num_code, config.data.win, config.data.hop)
        self.peq = ParametricEqualizer(
            config.data.sr, config.data.win)
        self.register_buffer(
            'window',
            torch.hann_window(config.data.win),
            persistent=False)
        f_min, f_max, peaks = \
            config.train.cutoff_lowpass, \
            config.train.cutoff_highpass, config.train.num_peak
        self.register_buffer(
            'peak_centers',
            f_min * (f_max / f_min) ** (torch.arange(peaks) / (peaks - 1)),
            persistent=False)

    def forward(self,
                wavs: torch.Tensor,
                pitch_shift: Optional[torch.Tensor] = None,
                formant_shift: Optional[torch.Tensor] = None,
                quality_power: Optional[torch.Tensor] = None,
                gain: Optional[torch.Tensor] = None,
                mode: str = 'linear',
                return_aux: bool = False) -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            formant_shift: [torch.float32; [B]], formant shifts.
            quality_power: [torch.float32; [B, num_peak + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peak]], gain in decibel.
            mode: interpolation mode, `linear` or `nearest`.
            return_aux: return the auxiliary values if True, only for debugging purpose.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        auxs = {}
        # [B, F, T / S], complex64
        fft = torch.stft(
            wavs,
            self.config.data.fft,
            self.config.data.hop,
            self.config.data.win,
            self.window,
            return_complex=True)
        # PEQ
        if quality_power is not None:
            # alias
            q_min, q_max = self.config.train.q_min, self.config.train.q_max
            # [B, num_peak + 2]
            q = q_min * (q_max / q_min) ** quality_power
            if gain is None:
                # [B, num_peak]
                gain = torch.zeros_like(q[:, :-2])
            # B
            bsize, _ = wavs.shape
            # [B, num_peak]
            center = self.peak_centers[None].repeat(bsize, 1)
            # [B, F]
            peaks = torch.prod(
                self.peq.peaking_equalizer(center, gain, q[:, :-2]), dim=1)
            # [B, F]
            lowpass = self.peq.low_shelving(
                self.config.train.cutoff_lowpass, q[:, -2])
            highpass = self.peq.high_shelving(
                self.config.train.cutoff_highpass, q[:, -1])
            # [B, F]
            filters = peaks * highpass * lowpass
            # [B, F, T / S]
            fft = fft * filters[..., None]
            # debugging purpose
            auxs.update({'peaks': peaks, 'highpass': highpass, 'lowpass': lowpass})
        # random formant, pitch shifter
        if formant_shift is not None or pitch_shift is not None:
            # [B, T / S, num_code], normalize the fft values for accurate LPC analysis
            code = self.coder.from_stft(fft / fft.abs().mean(dim=1)[:, None].clamp_min(1e-7))
            # [B, T / S, F]
            filter_ = self.coder.envelope(code)
            source = fft.transpose(1, 2) / (filter_ + 1e-7)
            # for debugging purpose
            auxs.update({'code': code, 'filter': filter_, 'source': source})
            # [B, T / S, F]
            if formant_shift is not None:
                filter_ = self.interp(filter_, formant_shift, mode=mode)
            if pitch_shift is not None:
                source = self.interp(source, pitch_shift, mode=mode)
            # [B, F, T / S]
            fft = (source * filter_).transpose(1, 2)
            # debugging purpose
            auxs.update({'ifilter': filter_, 'isource': source})
        # [B, T]
        out = torch.istft(
            fft,
            self.config.data.fft,
            self.config.data.hop,
            self.config.data.win,
            self.window)
        # max value normalization
        out = out / out.max(dim=-1, keepdim=True).values.clamp_min(1e-7)
        if return_aux:
            return out, auxs
        return out

    @staticmethod
    def complex_interp(inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Interpolate the complex tensor.
        Args:
            inputs: [torch.complex64; [B, C, ...]], complex inputs.
        Returns:
            [torch.complex64; [B, C, ...]], interpolated.
        """
        mag = F.interpolate(inputs.abs(), *args, **kwargs)
        angle = F.interpolate(inputs.angle(), *args, **kwargs)
        return torch.polar(mag, angle)

    def interp(self,
               inputs: torch.Tensor,
               shifts: torch.Tensor,
               mode: str) -> torch.Tensor:
        """Interpolate the channel axis with dynamic shifts.
        Args:
            inputs: [torch.complex64; [B, T, C]], input tensor.
            shifts: [torch.float32; [B]], shift factor.
            mode: interpolation mode.
        Returns:
            [torch.complex64; [B, T, C]], interpolated.
        """
        INTERPOLATION = {
            torch.float32: F.interpolate,
            torch.complex64: Augment.complex_interp}
        assert inputs.dtype in INTERPOLATION, 'unsupported interpolation'
        interp_fn = INTERPOLATION[inputs.dtype]
        # _, _, C
        _, _, channels = inputs.shape
        # B x [1, T, C x min(1., shifts)]
        interp = [
            interp_fn(
                f[None], scale_factor=s.item(), mode=mode)[..., :channels]
            for f, s in zip(inputs, shifts)]
        # [B, T, C]
        return torch.cat([
            F.pad(f, [0, channels - f.shape[-1]])
            for f in interp], dim=0)
