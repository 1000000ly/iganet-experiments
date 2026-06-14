"""
FNO2d — Fourier Neural Operator (2D)
Reference: Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations", ICLR 2021.

Weights stored as real [in, out, modes, modes, 2] (last dim = real/imag) so that
numel() counts are directly comparable to E1's ~580k float64 parameters.
With width=16, modes=12, n_layers=4: total ~593k parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cmul2d(inp, weights):
    """Complex multiply: inp [B, in, m1, m2] × weights [in, out, m1, m2, 2] → [B, out, m1, m2].

    Weights are stored as real tensors with last dim (real, imag).
    Works on CPU, MPS, and CUDA without needing complex tensor support.
    """
    w_r = weights[..., 0]
    w_i = weights[..., 1]
    inp_r, inp_i = inp.real, inp.imag
    out_r = (torch.einsum("bixy,ioxy->boxy", inp_r, w_r)
             - torch.einsum("bixy,ioxy->boxy", inp_i, w_i))
    out_i = (torch.einsum("bixy,ioxy->boxy", inp_r, w_i)
             + torch.einsum("bixy,ioxy->boxy", inp_i, w_r))
    return torch.complex(out_r, out_i)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Shape: [in, out, m1, m2, 2]  — numel = in*out*m1*m2*2 real params per tensor
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)           # [B, C, H, W//2+1] complex

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )
        m1, m2 = self.modes1, self.modes2

        # Top-left quadrant of frequency space
        out_ft[:, :, :m1, :m2] = _cmul2d(
            x_ft[:, :, :m1, :m2], self.weights1
        )
        # Bottom-left quadrant (negative frequencies in first dim)
        out_ft[:, :, -m1:, :m2] = _cmul2d(
            x_ft[:, :, -m1:, :m2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W))   # [B, out, H, W]


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Input:  [B, H, W, 2]  (G_grid: physical x/y on parametric grid)
    Output: [B, H, W]     (predicted u field)
    """

    def __init__(self, modes1: int = 12, modes2: int = 12,
                 width: int = 16, n_layers: int = 4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(2, width)

        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, 2]
        x = self.fc0(x)              # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)   # [B, width, H, W]

        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))

        x = x.permute(0, 2, 3, 1)   # [B, H, W, width]
        x = F.gelu(self.fc1(x))      # [B, H, W, 128]
        x = self.fc2(x)              # [B, H, W, 1]
        return x.squeeze(-1)         # [B, H, W]
