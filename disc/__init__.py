from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config
from .mpd import MultiPeriodDiscriminator


class Discriminator(nn.Module):
    """NANSY++ discriminator.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: discriminator hyperparameters.
        """
        super().__init__()
        self.config = config
        self.mpd = MultiPeriodDiscriminator(
            config.channels,
            config.periods,
            config.kernels,
            config.strides,
            config.postkernels,
            config.leak)

    def forward(self, inputs: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Discriminates whether inputs are real of synthesized.
        Args:
            inputs: [torch.float32; [B, T]], input signal.
        Returns:
            [torch.float32; [B, S]], list of logits.
            [torch.float32; [B, C, F, S]], feature maps.
        """
        return self.mpd(inputs)

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
        disc = cls(config)
        disc.load_(states, optim)
        return disc
