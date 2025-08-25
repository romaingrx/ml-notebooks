# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import torch
import torchvision
from einops import rearrange
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


class Encoder(nn.Module):
    def __init__(self, *, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.log_sigma = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, *, latent_dim: int, output_dim: int = 784):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.conv1T = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2T = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = rearrange(x, "b (c h w) -> b c h w", c=64, h=7, w=7)
        x = torch.relu(self.conv1T(x))
        x = torch.sigmoid(self.conv2T(x))
        return x.view(x.size(0), self.output_dim)


class VAE(nn.Module):
    def __init__(self, *, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encoder(x)
        z = VAE.reparametrize(mu, log_sigma)
        return self.decoder(z).view(x.shape), mu, log_sigma

    @staticmethod
    def reparametrize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function for the VAE, see the README for more details.

        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of the latent space.
            log_sigma: Log of the standard deviation of the latent space.
        """
        BCE = (recon_x - x).pow(2).sum()
        KL = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return BCE + KL


# %%

vae = VAE(latent_dim=2).to(device)
print(f"Number of parameters: {sum(p.numel() for p in vae.parameters())}")

x, _ = next(iter(train_dataloader))
print(x.shape)

# %%


def train_epoch(
    model: VAE, dataloader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    model.train()
    total_loss = 0.0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, log_sigma = model(x)
        loss = model.loss_function(recon_x, x, mu, log_sigma)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# %%

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)


losses = []
for epoch in range(10):
    loss = train_epoch(vae, train_dataloader, optimizer)
    losses.append(loss)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# %%


plt.plot(losses)
plt.show()

# %%

# Let's generate an example from a random sample

z = torch.randn(1, 2).to(device)
x = vae.decoder(z).detach().cpu()
print(x.shape)

plt.imshow(x[0].numpy().reshape(28, 28), cmap="gray")
plt.show()

# %%
