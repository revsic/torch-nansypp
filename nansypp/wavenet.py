import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    """WaveNet block, dilated convolution and skip connection.
    """
    def __init__(self,
                 channels: int,
                 aux: int,
                 kernels: int,
                 dilations: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            aux: size of the auxiliary input channels.
            kernels: size of the convolutional kernel.
            dilations: dilation rates of contoluion.
        """
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                channels, channels * 2, kernels,
                padding=(kernels - 1) * dilations // 2, dilation=dilations))

        self.proj_aux = nn.utils.weight_norm(
            nn.Conv1d(aux, channels * 2, 1, bias=False))

        self.proj_res = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1))
        self.proj_skip = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1))

    def forward(self,
                inputs: torch.Tensor,
                aux: torch.Tensor) -> torch.Tensor:
        """Pass to the wavenet block.
        Args:
            inputs: [torch.float32; [B, channels, T]], input tensor.
            aux: [torch.float32; [B, aux, T]], auxiliary input tensors.
        Returns:
            residual: [torch.float32; [B, C, T]], residually connected.
            skip: [torch.float32; [B, C, T]], skip connection purposed.
        """
        # [B, C x 2, T]
        x = self.conv(inputs) + self.proj_mel(aux)
        # [B, C, T]
        gate, context = x.chunk(2, dim=1)
        # [B, C, T]
        x = torch.sigmoid(gate) * torch.tanh(context)
        # [B, C, T]
        res = (x + self.proj_res(x)) * (2 ** -0.5)
        return res, self.proj_skip(x)


class WaveNet(nn.Module):
    """WaveNet, Oord et al., 2016.
    """
    def __init__(self,
                 channels: int,
                 aux: int,
                 kernels: int,
                 dilation_rate: int,
                 layers: int,
                 cycles: int):
        """Initializer.
        Args:
            channels: size of the hidden channels.
            aux: size of the auxiliary input channels.
            kernels: size of the convolutional kernels.
            dilation_rate: dilaion rate.
            layers: the number of the wavenet blocks in single cycle.
            cycles: the number of the cycles.
        """
        super().__init__()
        # channels=64
        self.proj_signal = nn.utils.wegith_norm(nn.Conv1d(1, channels, 1))
        # aux=1024, cycles=3, layers=10, dilation_rate=2
        self.blocks = nn.ModuleList([
            WaveNetBlock(channels, aux, kernels, dilation_rate ** j)
            for _ in range(cycles)
            for j in range(layers)])

        self.proj_out = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(channels, 1, 1)),
            nn.Tanh())

    def forward(self, noise: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Generate the signal from noise and auxiliary inputs.
        Args:
            noise: [torch.float32; [B, T]], initial noise signal.
            aux: [torch.float32; [B, aux, T]], auxiliary inpus.
        Returns;
            [torch.float32; [B, T]], generated signal.
        """
        # [B, channels, T]
        x = self.proj_signal(noise[:, None])
        # (layers x cycles) x [B, channels, T]
        skips = 0.
        for block in self.blocks:
            # [B, channels, T], [B, channels, T]
            x, skip = block(x, aux)
            skips = skips + skip
        # [B, T]
        return self.proj_out(
            skips * (len(self.blocks) ** -0.5)).squeeze(dim=1)
