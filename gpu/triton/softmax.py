# %%

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from shared import get_logger

device = driver.active.get_active_torch_device()
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
    input_ptr: tl.tensor,
    input_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Consts
    MINUS_INF = float("-inf")
    output_stride = input_stride

    pid: int = tl.program_id(axis=0)  # Used to get which row we're tackling
    input_ptr_start: int = input_ptr + pid * input_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr_chunks = input_ptr_start + offsets
    mask = offsets < input_stride

    x: tl.tensor = tl.load(
        input_ptr_chunks, mask=mask, other=MINUS_INF
    )  # Load to SRAM and make sure we get only the number of cols max

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


# %% Stolen from https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#benchmark


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
            "naive_softmax",
        ],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax_fn(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True)
