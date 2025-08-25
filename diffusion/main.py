# %%
# %load_ext autoreload
# %autoreload 2

import math

import einops
import matplotlib.pyplot as plt
import torch
import torchvision
from diffusers import UNet2DModel
from torch import nn
from torch.utils.data import DataLoader

from diffusion.scheduler import LinearScheduler, SchedulerProtocol
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
def corrupt(
    x: torch.Tensor, t: int, scheduler: SchedulerProtocol
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt the input `x` by mixing it with noise according to `amount`
    Returns: (noisy_x, noise) - both the corrupted image and the noise that was added
    """
    noise = torch.randn_like(x)  # Use Gaussian noise, not uniform
    alpha_bar = scheduler.get_alpha_bar(t)
    noisy_x = x * math.sqrt(alpha_bar) + noise * math.sqrt(1 - alpha_bar)
    return noisy_x, noise


scheduler = LinearScheduler(beta_start=0.1, beta_end=0.9, timesteps=10)
x, y = next(iter(train_dataloader))

x_0 = x[0].to(device)
c, h, w = x_0.shape  # Fixed order: channels, height, width
x_ts = einops.rearrange(
    torch.stack([corrupt(x_0, i, scheduler)[0] for i in range(1, 10)], dim=0),
    "... h w -> h (... w)",
)

plt.figure(figsize=(15, 5))
plt.title("Image and Corrupted version")
plt.imshow(x_ts.cpu().squeeze(), cmap="gray")
plt.show()


# %% Training loop
batch_size = 64
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 20
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, timesteps)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alphas_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alphas_bars = torch.sqrt(1 - alpha_bars)


net = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # Reduce complexity
    block_out_channels=(16, 32, 64),  # 3 levels instead of 4
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    ),
    up_block_types=(
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
net.to(device)

loss_fn = nn.MSELoss()

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

losses: list[float] = []

# The training loop
for epoch in range(epochs):
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        imgs = x.to(device)
        batch_size = x.shape[0]
        noise = torch.randn_like(imgs, device=device)
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        noisy_imgs = (
            torch.sqrt(alpha_bars).to(device)[t, None, None, None] * imgs
            + torch.sqrt(1 - alpha_bars).to(device)[t, None, None, None] * noise
        )

        opt.zero_grad()
        pred_noise = net(noisy_imgs, t, return_dict=False)[0]

        loss = loss_fn(pred_noise, noise)

        loss.backward()
        opt.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

plt.plot(losses)
plt.ylim(0, 0.1)

# %%


sample_batch_size = 2

with torch.no_grad():
    x = torch.randn(sample_batch_size, c, h, w, device=device)
    for t in reversed(range(timesteps)):
        # Create timestep tensor for the batch
        timestep = torch.full((sample_batch_size,), t, device=device, dtype=torch.long)
        pred_noise = net(x, timestep).sample
        noise = torch.randn_like(x) if t > 0 else 0.0

        alpha = alphas[t]
        beta = betas[t]
        alpha_bar = alpha_bars[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        ) + torch.sqrt(beta) * noise


x_grid = einops.rearrange(x, "b c h w -> h (b c w)")

plt.imshow(x_grid.cpu().squeeze(), cmap="gray")
