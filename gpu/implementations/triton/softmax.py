# %%

import torch
import triton
import triton.language as tl

from shared import get_device, get_logger

device = get_device()
logger = get_logger(__name__)

logger.info(f"{device=}")
torch.random.manual_seed(42)

X = torch.randn((2, 4), device=device)
logger.info(f"{X=}")


# %%


def naive_softmax_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    z = x - x_max
    exp = z.exp()
    sum_exp = exp.sum(dim=dim, keepdim=True)
    return exp / sum_exp


@triton.jit
def _softmax_kernel(
    output_ptr: tl.tensor,
    output_stride: tl.constexpr,
    input_ptr: tl.tensor,
    input_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    MINUS_INF = float("-inf")
    pid: int = tl.program_id(axis=0)  # Used to get which row we're tackling
    input_ptr_start: int = input_ptr + pid * input_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr_chunks = input_ptr_start + offsets
    mask = offsets < input_stride

    x: tl.tensor = tl.load(
        input_ptr_chunks, mask=mask, other=MINUS_INF
    )  # Load to SRAM and make sure we get only the number of cols

    # Actual softmax
    z = x - tl.max(x, axis=0)
    exp = tl.exp(z)
    sm_out = exp / tl.sum(exp, axis=0)

    output_ptr_start = output_ptr + pid * output_stride
    output_ptr_chunks = output_ptr_start + offsets
    tl.store(output_ptr_chunks, sm_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Only accept 2D array for now"
    rows, cols = x.shape
    block_size = triton.next_power_of_2(cols)
    num_warps = 4  # * 32 per wrap
    grid = (rows,)

    sm_out = torch.empty_like(x)
    _softmax_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return sm_out


torch_softmax = torch.nn.functional.softmax(X, dim=-1)
naive_softmax = naive_softmax_fn(X)
logger.info(
    f"Equivalence torch vs naive: {torch.allclose(torch_softmax, naive_softmax)}"
)
triton_softmax = softmax(X)
logger.info(
    f"Equivalence torch vs triton: {torch.allclose(torch_softmax, triton_softmax)}"
)


# %%

X.stride(0)
