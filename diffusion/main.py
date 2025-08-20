# %%
# %load_ext autoreload
# %autoreload 2

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from shared import TORCH_DATA_DIR, get_device

device = get_device()
print(f"Using device: {device}")

# %%

dataset = torchvision.datasets.MNIST(
    root=TORCH_DATA_DIR / "mnist",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# %%
def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
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

    def forward(self, x: torch.Tensor):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


# %% Util cell
net = BasicUNet()
x = torch.rand(8, 1, 28, 28)
net(x).shape

sum([p.numel() for p in net.parameters()])
