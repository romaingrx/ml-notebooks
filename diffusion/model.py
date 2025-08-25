from typing import Protocol

import torch
from torch import nn


class DiffusionModelProtocol(Protocol):
    def __init__(self, in_channels: int, out_channels: int):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BasicUNet(nn.Module, DiffusionModelProtocol):
    """A minimal UNet implementation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, layer in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(layer(x))  # Through the layer and the activation function

        return x
