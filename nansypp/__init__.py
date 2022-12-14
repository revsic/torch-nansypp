from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

from .transform import ConstantQTransform, MelSpectrogram
from .framelevel import FrameLevelSynthesizer
from .linguistic import LinguisticEncoder
from .pitch import PitchEncoder
from .synthesizer import Synthesizer
from .timber import TimberEncoder
from .wav2vec2 import Wav2Vec2Wrapper


class Nansypp(nn.Module):
    """NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: NANSY++ configurations.
        """
        super().__init__()
        self.config = config
        # assume the output channels of wav2vec2.forward is `config.w2v2_channels`
        self.wav2vec2 = Wav2Vec2Wrapper(config.w2v2_name, config.sr, config.w2v2_lin)
        self.linguistic = LinguisticEncoder(
            config.w2v2_channels,
            config.ling_hiddens,
            config.ling_preconv,
            config.ling_kernels,
            config.leak,
            config.dropout)

        self.cqt = ConstantQTransform(
            config.cqt_hop,
            config.cqt_fmin,
            config.cqt_fmax,
            config.cqt_bins,
            config.cqt_bins_per_octave,
            config.sr)

        self.pitch = PitchEncoder(
            config.pitch_freq,
            config.pitch_prekernels,
            config.pitch_kernels,
            config.pitch_channels,
            config.pitch_blocks,
            config.pitch_gru,
            config.pitch_hiddens,
            config.pitch_f0_bins)
        
        self.register_buffer(
            'pitch_bins', 
            # linear space in log-scale
            torch.linspace(
                np.log(config.pitch_start),
                np.log(config.pitch_end),
                config.pitch_f0_bins).exp())

        self.melspec = MelSpectrogram(
            config.mel_hop,
            config.mel_win,
            config.mel,
            config.mel_fmin,
            config.mel_fmax,
            sr=config.sr)

        self.timber = TimberEncoder(
            config.mel,
            config.timb_global,
            config.timb_channels,
            config.timb_prekernels,
            config.timb_scale,
            config.timb_kernels,
            config.timb_dilations,
            config.timb_bottleneck,
            config.timb_hiddens,
            config.timb_latent,
            config.timb_timber,
            config.timb_tokens,
            config.timb_heads,
            # [f0, Ap, Aap, L, g]
            config.ling_hiddens + config.timb_global + 3,
            config.timb_slerp)

        self.framelevel = FrameLevelSynthesizer(
            config.ling_hiddens,
            config.timb_global,
            config.frame_kernels,
            config.frame_dilations,
            config.frame_blocks,
            config.leak,
            config.dropout)

        self.synthesizer = Synthesizer(
            config.cqt_hop,
            config.sr,
            config.synth_channels,
            config.ling_hiddens,
            config.synth_kernels,
            config.synth_dilation_rate,
            config.synth_layers,
            config.synth_cycles)

    def analyze_pitch(self, inputs: torch.Tensor, index: int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the pitch and periodical, aperiodical amplitudes.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
            index: CQT start index.
        Returns:
            [torch.float32; []], CQT features.
            [torch.float2; [B, N]], frame-level pitch and amplitude sequence.
        """
        # [B, cqt_bins, N(=T / cqt_hop)]
        cqt = self.cqt(inputs)
        # alias
        freq = self.config.pitch_freq
        # [B, N, f0_bins], [B, N], [B, N]
        pitch_bins, p_amp, ap_amp = self.pitch.forward(cqt[:, index:index + freq])
        # [B, N]
        pitch = (pitch_bins * self.pitch_bins).sum(dim=-1)
        # [], [B, N]
        return cqt, pitch, p_amp, ap_amp
    
    def analyze_linguistic(self, inputs: torch.Tensor) -> torch.Tensor:
        """Analyze the linguistic informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, ling_hiddens, S]], linguistic informations.
        """
        # [B, S, w2v2_channels]
        w2v2 = self.wav2vec2.forward(inputs)
        # [B, ling_hiddens, S]
        return self.linguistic.forward(w2v2.transpose(1, 2))

    def analyze_timber(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze the timber informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, timb_global]], global timber emebddings.
            [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
        """
        # [B, mel, T / mel_hop]
        mel = self.melspec.forward(inputs)
        # [B, timb_global], [B, timb_timber, timb_tokens]
        return self.timber.forward(mel)

    def analyze(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze the input signal.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns;
            analyzed featuers,
                cqt: [torch.float32; []], CQT features.
                pitch, p_amp, ap_amp: [torch.float2; [B, N]],
                    frame-level pitch and amplitude sequence.
                ling: [torch.float32; [B, ling_hiddens, S]], linguistic informations.
                timber_global: [torch.float32; [B, timb_global]], global timber emebddings.
                timber_bank: [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
        """
        # [], [B, N]
        cqt, pitch, p_amp, ap_amp = self.analyze_pitch(inputs)
        # [B, ling_hiddens, S]
        ling = self.analyze_linguistic(inputs)
        # [B, timb_global], [B, timb_timber, timb_tokens]
        timber_global, timber_bank = self.analyze_timber(inputs)
        return {
            'cqt': cqt,
            'pitch': pitch,
            'p_amp': p_amp,
            'ap_amp': ap_amp,
            'ling': ling,
            'timber_global': timber_global,
            'timber_bank': timber_bank}

    def synthesize(self,
                   pitch: torch.Tensor,
                   p_amp: torch.Tensor,
                   ap_amp: torch.Tensor,
                   ling: torch.Tensor,
                   timber_global: torch.Tensor,
                   timber_bank: torch.Tensor,
                   noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize the signal.
        Args:
            pitch, p_amp, ap_amp: [torch.float32; [B, N]], frame-level pitch, amplitude sequence.
            ling: [torch.float32; [B, ling_hiddens, S]], linguistic features.
            timber_global: [torch.float32; [B, timb_global]], global timber.
            timber_bank: [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation and synthesized speech signal.
        """
        # S
        ling_len = ling.shape[-1]
        # [B, 3, S]
        pitch_rel = F.interpolate(torch.stack([pitch, p_amp, ap_amp], dim=1), size=ling_len)
        # [B, 3 + ling_hiddens + timb_global, S]
        contents = torch.cat([
            pitch_rel, ling, timber_global[..., None].repeat(1, 1, ling_len)], dim=1)
        # [B, timber_global, S]
        timber_sampled = self.timber.sample_timber(contents, timber_global, timber_bank)
        # [B, ling_hiddens, S]
        frame = self.framelevel.forward(ling, timber_sampled)
        # [B, T], [B, T]
        return self.synthesizer.forward(pitch, p_amp, ap_amp, frame, noise)

    def forward(self, inputs: torch.Tensor, noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct input audio.
        Args:
            inputs: [torch.float32; [B, T]], input signal.
            noise: [torch.float32; [B, T]], predefined noise for excitation, if provided.
        Returns:
            [torch.float32; [B, T]], reconstructed.
            auxiliary outputs, reference `Nansypp.analyze`.
        """
        features = self.analyze(inputs)
        # [B, T]
        excitation, synth = self.synthesize(
            features['pitch'],
            features['p_amp'],
            features['ap_amp'],
            features['ling'],
            features['timber_global'],
            features['timber_bank'],
            noise=noise)
        # update
        features['excitation'] = excitation
        return synth, features

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict(), 'config': vars(self.config)}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load_(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints inplace.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])

    @classmethod
    def load(cls, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        config = Config()
        for key, val in states['config'].items():
            if not hasattr(config, key):
                import warnings
                warnings.warn(f'unidentified key {key}')
                continue
            setattr(config, key, val)
        # construct
        nansypp = cls(config)
        nansypp.load_(states, optim)
        return nansypp
