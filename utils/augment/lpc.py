import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPredictiveCoding(nn.Module):
    """LPC: Linear-predictive coding supports.
    """
    def __init__(self, num_code: int, windows: int, strides: int):
        """Initializer.
        Args:
            num_code: the number of the coefficients.
            windows: size of the windows.
            strides: the number of the frames between adjacent windows.
        """
        super().__init__()
        self.num_code = num_code
        self.windows = windows
        self.strides = strides

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the linear-predictive coefficients from inputs.
        Args:
            inputs: [torch.float32; [B, T]], audio signal.
        Returns:
            [torch.float32; [B, T / strides, num_code]], coefficients.
        """
        # alias
        w = self.windows
        # [B, T / strides, windows]
        frames = F.pad(inputs, [0, w]).unfold(-1, w, self.strides)
        # [B, T / strides, windows]
        corrcoef = LinearPredictiveCoding.autocorr(frames)
        # [B, T / strides, num_code]
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[..., :self.num_code + 1])

    def from_stft(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the linear-predictive coefficients from STFT.
        Args:
            inputs: [torch.complex64; [B, windows // 2 + 1, T / strides]], fourier features.
        Returns:
            [torch.float32; [B, T / strides, num_code]], linear-predictive coefficient.
        """
        # [B, windows, T / strides]
        corrcoef = torch.fft.irfft(inputs.abs().square(), dim=1)
        # [B, num_code, T / strides]
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[:, :self.num_code + 1].transpose(1, 2))

    def envelope(self, lpc: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """LPC to spectral envelope.
        Args:
            lpc: [torch.float32; [..., num_code]], coefficients.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], filters.
        """
        denom = torch.fft.rfft(
            -F.pad(lpc, [1, 0], value=1.), self.windows, dim=-1).abs()
        return denom.clamp_min(eps) ** -1

    @staticmethod
    def autocorr(wavs: torch.Tensor) -> torch.Tensor:
        """Compute the autocorrelation.
        Args:
            wavs: [torch.float32; [..., T]], audio signal.
        Returns:
            [torch.float32; [..., T]], auto-correlation.
        """
        # [..., T // 2 + 1], complex64
        fft = torch.fft.rfft(wavs, dim=-1)
        # [..., T]
        return torch.fft.irfft(fft.abs().square(), dim=-1)

    @staticmethod
    def solve_toeplitz(corrcoef: torch.Tensor) -> torch.Tensor:
        """Solve the toeplitz matrix.
        Args:
            corrcoef: [torch.float32; [..., num_code + 1]], auto-correlation.
        Returns:
            [torch.float32; [..., num_code]], solutions.
        """
        ## solve the first row
        # [..., 2]
        solutions = F.pad(
            (-corrcoef[..., 1] / corrcoef[..., 0].clamp_min(1e-7))[..., None],
            [1, 0], value=1.)
        # [...]
        extra = corrcoef[..., 0] + corrcoef[..., 1] * solutions[..., 1]
        
        ## solve residuals
        num_code = corrcoef.shape[-1] - 1
        for k in range(1, num_code):
            # [...]
            lambda_value = (
                    -solutions[..., :k + 1]
                    * torch.flip(corrcoef[..., 1:k + 2], dims=[-1])
                ).sum(dim=-1) / extra.clamp_min(1e-7)
            # [..., k + 2]
            aug = F.pad(solutions, [0, 1])
            # [..., k + 2]
            solutions = aug + lambda_value[..., None] * torch.flip(aug, dims=[-1])
            # [...]
            extra = (1. - lambda_value ** 2) * extra
        # [..., num_code]
        return solutions[..., 1:]
