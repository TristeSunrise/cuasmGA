import argparse
from dataclasses import dataclass, field
from typing import Optional

import os
import random
import numpy as np
import torch

import triton
import triton.language as tl

from jit import jit
from autotuner import autotune as fgk_autotune
from gpu_utils import get_gpu_name
from autotuner import triton_autotune_with_cache

@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: bool = False
    tt: bool = False

    # Workload
    m: int = 1024
    n: int =16384
    bm: int = 8
    bn: int = 64

    gpu: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=4096)
    parser.add_argument("--bm", type=int, default=16)
    parser.add_argument("--bn", type=int, default=128)

    args = parser.parse_args()
    config = Config(**vars(args))
    return config

GPU = get_gpu_name()


@triton.jit
def tt_kernel(output_ptr, input_ptr,
                   input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   ):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


# device = torch.cuda.current_device()
# properties = driver.active.utils.get_device_properties(device)
# NUM_SM = properties["multiprocessor_count"]
# NUM_REGS = properties["max_num_regs"]
# SIZE_SMEM = properties["max_shared_mem"]
# WARP_SIZE = properties["warpSize"]
# target = triton.runtime.driver.active.get_current_target()
# kernels = {}


def tt_call(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    # num_stages = 4 if SIZE_SMEM > 200000 else 2
    num_stages = 4

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    # kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    # if kernel is None:
    #     kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
    #                                    num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    #     kernel._init_handles()
    #     n_regs = kernel.n_regs
    #     size_smem = kernel.metadata.shared
    #     occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    #     occupancy = min(occupancy, SIZE_SMEM // size_smem)
    #     num_programs = NUM_SM * occupancy
    #     kernels[BLOCK_SIZE] = (kernel, num_programs)

    # num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    # kernel[(num_programs, 1, 1)](
    #     y,
    #     x,
    #     x.stride(0),
    #     y.stride(0),
    #     n_rows,
    #     n_cols,
    # )
    tt_kernel[(n_rows, )] (
        y, x, x.stride(0), y.stride(0),
        n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y

def call(x, load_dir, kernel):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # num_warps = 8
    # num_stages = 4

    # Allocate output
    y = torch.empty_like(x)

    kernel[(n_rows, )] (
        y, x, x.stride(0), y.stride(0),
        n_rows, n_cols,
        #BLOCK_SIZE=BLOCK_SIZE,
        # num_stages=num_stages,
        load_dir = load_dir,
    )
    return y

def main():
    args = parse_args()

    # %%
    # Unit Test
    # ---------
    M, N = args.m , args.n
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # x = torch.randn(M, N, device='cuda')
    # y_triton = tt_call(x)
    # y_torch = torch.softmax(x, axis=1)
    # assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    # print("✅ Triton and Torch match")

    device = torch.device("cuda:0")
    dtype=torch.float32

    M, N = args.m , args.n
    BLOCK_M, BLOCK_N = args.bm, args.bn
    x = torch.randn((M, N), dtype=dtype, device=device)
    BLOCK_SIZE = triton.next_power_of_2(N)

    args.total_flops = 2 * x.nelement() * x.element_size()
    # args.save_dir = f'{GPU}/persistent_softmax/{M}_{N}_{BLOCK_M}_{BLOCK_N}'
    args.save_dir = f'{GPU}/persistent_softmax/{M}_{N}'
    if args.load is None:
        load_dir = None
    elif args.load == "auto":
        load_dir = f'{args.default_out_path}/{GPU}/persistent_softmax/{M}_{N}'
    else:
        load_dir = args.load

    @fgk_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_stages=4, num_warps=8),
        ],
        key=['n_rows', 'n_cols'],
        ga_config=args,
        ret_ptr=0,
    )
    @jit
    def ga_kernel(output_ptr, input_ptr,
                    input_row_stride, output_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr,
        ):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    out = call(x, load_dir, ga_kernel)
    if args.tt:
        ref = tt_call(x)
        assert torch.allclose(out, ref)
    print("✅ out match")


    # %%
    # Benchmark
    # ---------
    if not args.bench:
        return

    print(f'benchmarking: {M=}; {N=}')
    torch.cuda.synchronize()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['NA'],  # argument names to use as an x-axis for the plot
            #x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
            x_vals=[0],
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot

            line_vals=['triton', 'torch', 'ga'],
            line_names=[ "Triton", "Torch", 'ga', ],
            # line_vals=['triton', 'torch', ],
            # line_names=[ "Triton", "Torch", ],

            styles=[('blue', '-'), ('green', '-'), ('red', '-')],
            ylabel="GB/s",  # label name for the y-axis
            plot_name="softmax-performance",
            #args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
            args={},
        ))
    def benchmark(NA, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), warmup=100, rep=100)
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: tt_call(x), warmup=100, rep=100)
        if provider == 'ga':
            ms = triton.testing.do_bench(lambda: call(x, load_dir, ga_kernel), warmup=100, rep=100)
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=True, print_data=True)

if __name__ == '__main__':
    main()
