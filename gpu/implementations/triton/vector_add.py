# %%
import logging

import torch
import triton
import triton.language as tl

from shared import get_device

device = get_device()
logger = logging.getLogger(__name__)
logger.info(f"{device=:}")
torch.random.manual_seed(42)

A = torch.randn((1, 4), device=device)
B = torch.randn((1, 4), device=device)

# %%


@triton.jit
def _vector_add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid: int = tl.program_id(axis=0)
    block_start: int = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    assert x.device == y.device == output.device
    assert x.dim() == y.dim() == 1
    assert x.shape[0] == y.shape[0]

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


x = torch.tensor([0, 1, 2, 3, 4, 5] * 1000000, device=device)
y = torch.tensor([0, 1, 2, 3, 4, 5] * 1000000, device=device)
vector_add(x, y)

# %%
