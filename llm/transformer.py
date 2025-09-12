# %%
import logging
import math

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, *, n_heads: int, dim_model: int, dim_kq: int, dim_v: int):
        super().__init__()
        self._n_heads: int = n_heads
        self._dim_model: int = dim_model
        self._dim_kq: int = dim_kq
        self._dim_v: int = dim_v

        self.Q = nn.Linear(self._dim_model, self._dim_kq * self._n_heads)
        self.K = nn.Linear(self._dim_model, self._dim_kq * self._n_heads)
        self.V = nn.Linear(self._dim_model, self._dim_v * self._n_heads)
        self.O = nn.Linear(self._dim_v * self._n_heads, self._dim_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x: torch.Tensor[B, dim_ctx, dim_model]
        """
        queries: torch.Tensor = einops.rearrange(
            self.Q(x),
            "B T (h d) -> B T h d",
            h=self._n_heads,
        )
        keys: torch.Tensor = einops.rearrange(
            self.K(x),
            "B T (h d) -> B T h d",
            h=self._n_heads,
        )
        values: torch.Tensor = einops.rearrange(
            self.V(x),
            "B T (h d) -> B T h d",
            h=self._n_heads,
        )

        raw_scores = einops.einsum(queries, keys, "B Tq h d, B Tk h d -> B h Tk Tq")
        if mask is not None:
            raw_scores = raw_scores.where(mask, float("-inf"))

        attention_scores = (raw_scores / math.sqrt(self._dim_kq)).softmax(-1)

        attention = einops.einsum(
            attention_scores, values, "B h Tk Tq, B Tq h dv -> B Tq h dv"
        )

        out = self.O(einops.rearrange(attention, "B T h d -> B T (h d)"))
        return out


class CausalMultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=0,
        )[None, :, :]
        logger.debug(f"mask shape: {causal_mask.shape}, mask: {causal_mask}")
        return super().forward(x, causal_mask)


# %%

B = 1
dim_model = 16
dim_ctx = 3

x = torch.rand((B, dim_ctx, dim_model))
mha = CausalMultiHeadSelfAttention(n_heads=3, dim_model=dim_model, dim_kq=4, dim_v=8)

output: torch.Tensor = mha(x)
logger.debug(output.shape)
