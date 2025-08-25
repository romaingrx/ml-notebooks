from typing import Protocol, final

import torch


class SchedulerProtocol(Protocol):
    def get_beta(self, t: int) -> float:
        pass

    def get_alpha(self, t: int) -> float:
        pass

    def get_alpha_bar(self, t: int) -> float:
        pass


@final
class LinearScheduler(SchedulerProtocol):
    beta_start: float
    beta_end: float
    timesteps: int

    def __init__(self, beta_start: float, beta_end: float, timesteps: int) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self._betas = torch.linspace(beta_start, beta_end, timesteps)

    def get_beta(self, t: int) -> float:
        return self._betas[t]

    def get_alpha(self, t: int) -> float:
        return 1 - self.get_beta(t)

    def get_alpha_bar(self, t: int) -> float:
        return torch.cumprod(1 - self._betas, dim=0, dtype=torch.float32)[t].item()
