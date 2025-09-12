# %%
# %load_ext autoreload
# %autoreload 2

import random
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import NamedTuple, cast

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch._tensor import Tensor
from tqdm import tqdm

from shared import get_device

device = get_device()
print(f"Using device: {device}")

# %%

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    pass

plt.ion()

# %%


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor | None
    reward: torch.Tensor


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# %%


class DQN(nn.Module):
    def __init__(self, *, in_features: int, n_actions: int):
        super(DQN, self).__init__()
        self.in_features: int = in_features
        self.n_actions: int = n_actions
        self.fc1: nn.Linear = nn.Linear(self.in_features, 32)
        self.fc2: nn.Linear = nn.Linear(32, 32)
        self.fc3: nn.Linear = nn.Linear(32, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state: torch.Tensor):
        if state.dim == 1:
            state = state[None]
        q_values = cast(torch.Tensor, self(state))
        action = q_values.argmax(-1, keepdim=True)
        return action


# %%

BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.1
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
N_EPISODES = 300


state, info = env.reset()

n_actions = 2
n_observations = len(state)
print(f"n_actions ({n_actions}) - n_observations ({n_observations})")
policy_net = DQN(in_features=n_observations, n_actions=n_actions).to(device)
target_net = DQN(in_features=n_observations, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

print(
    f"Output shape: {policy_net(torch.randn(1, n_observations, device=device)).shape}"
)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# %%

epsilon = 1.0
memory = ReplayMemory(MEMORY_CAPACITY)


def select_action(state: torch.Tensor) -> torch.Tensor:
    global epsilon
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net.act(state)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def obs_to_state(obs: torch.Tensor) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def optim_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions: list[Transition] = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def soft_update():
    policy_state = policy_net.state_dict()
    target_state = target_net.state_dict()

    for key in policy_state:
        target_state[key] = (1 - TAU) * target_state[key] + TAU * policy_state[key]

    target_net.load_state_dict(target_state)


@dataclass
class Tracker:
    timesteps: list[int] = field(default_factory=list)


tracker = Tracker()

trange = tqdm(range(N_EPISODES), desc="Episodes")
for episode in trange:
    trange.set_postfix({"timesteps": np.mean(tracker.timesteps[-50:])})

    obs, info = env.reset()
    state: Tensor = obs_to_state(obs)

    for t in tqdm(count(), desc="Timesteps", leave=False):
        action = select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = obs_to_state(obs) if not terminated else None

        memory.push(Transition(state, action, next_state, reward))

        optim_model()
        soft_update()

        state = next_state

        if done:
            epsilon = max(MIN_EPSILON, EPSILON_DECAY * epsilon)
            tracker.timesteps.append(t)
            break

plt.plot(tracker.timesteps)
plt.show()

# %%

env = gym.make("CartPole-v1", render_mode="human")
for _ in range(5):
    obs, info = env.reset()
    state = obs_to_state(obs)
    done = False

    while not done:
        env.render()
        action = target_net.act(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        state = obs_to_state(obs) if not terminated else None
        done = terminated or truncated

env.close()
