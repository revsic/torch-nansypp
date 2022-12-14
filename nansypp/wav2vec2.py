from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model


class Wav2Vec2Wrapper(nn.Module):
    """Wrapping huggingface wav2vec2.0.
    """
    DEFAULT = 'facebook/wav2vec2-large-xlsr-53'
    # Since 0-th hidden state is poosition-informed convolution features
    # , one-base indexing required
    # Default 12 for NANSY, 15 for NANSY++
    LINGUISTIC = 15

    OUT_CHANNELS = 1024

    def __init__(self,
                 name: Optional[str] = None,
                 sr: int = 16000,
                 linguistic: Optional[int] = None):
        """Load the wav2vec2.0 pretrained model.
        Args:
            name: name of the model, default use facebook XLSR-53.
            sr: sample rates of the input audio, default 16khz for XLSR-53.
            linguistic: layer outputs for linguistic features. 
        """
        super().__init__()
        name = name or Wav2Vec2Wrapper.DEFAULT
        # warning can occurs since `Wav2Vec2Model` does not contain
        # quantization modules
        self.model = Wav2Vec2Model.from_pretrained(name)

        self.sr = sr
        self.resample = torchaudio.transforms.Resample(sr, 16000)

        self.linguistic = linguistic or Wav2Vec2Wrapper.LINGUISTIC
        self.eval()

    @torch.no_grad()
    def forward(self,
                audio: torch.Tensor,
                audiolen: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T']], audio, [-1, 1]-ranged.
            audiolen: [torch.long; [B]], length of the audios,
                masking the inputs if provided.
        Returns:
            linguistic: [torch.float32; [B, S, C]], linguistic encodings,
                where S = T // 320, T = floor(T' / `sr` x 16000)
        """
        # [B, T]
        audio = self.resample(audio)
        # B, T
        bsize, timestep = audio.shape
        if audiolen is None:
            audiolen = torch.full(
                (bsize,), timestep, dtype=torch.long, device=audio.device)
        else:
            # rearange to 16khz audio frames
            audiolen = torch.ceil(audiolen / self.sr * 16000).to(torch.long)
        # [B, T]
        mask = (
            torch.arange(timestep, device=audiolen.device)[None]
            < audiolen[:, None]).to(torch.float32)
        ## normalize the inputs before feed to wav2vec2
        ## , reference Wav2VecFeatureExtractor
        # [B]
        mean = (audio * mask).sum(dim=-1) / audiolen.to(torch.float32)
        # [B]
        var = ((audio - mean[:, None]) * mask).square().sum(dim=-1) / audiolen.to(torch.float32)
        # [B, T], for numerical stability of square root
        normed = (audio - mean[:, None]) / (var[:, None] + 1e-7).sqrt()
        output = self.model(
            normed,
            attention_mask=mask.to(torch.long),
            output_hidden_states=True)
        # [B, S, C(=1024)]
        return output.hidden_states[self.linguistic]

    def train(self, mode: bool = True):
        """Support only evaluation
        """
        if mode:
            import warnings
            warnings.warn('WhisperWrapper does not support training mode')
        else:
            # super call
            super().train(False)

    def load_state_dict(self,
                        state_dict: Dict[str, torch.Tensor],
                        strict: bool = True):
        """Do not load state dict.
        """
        pass
