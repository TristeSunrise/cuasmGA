import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

from jit import jit
from autotuner import autotune as fgk_autotune
from gpu_utils import get_gpu_name, get_gpu_cc

from autotuner import triton_autotune_with_cache


# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: int = 0
    tt: bool = False

    # Workload
    Z: int = 1
    H: int = 4
    wl: int = 16384
    D_HEAD: int = 64

    gpu: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--Z", type=int, dest="Z", default=1)
    parser.add_argument("--H", type=int, dest="H", default=32)
    parser.add_argument("--wl", type=int, default=4096)
    parser.add_argument("--dh", type=int, dest="D_HEAD", default=64)

    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    config = Config(**vars(args))
    return config


GPU = get_gpu_name()



@triton.jit
def rmsnorm_tt(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def call_tt(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_tt[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=1e-6, BLOCK_N_SIZE=32,
                                num_stages=2, num_warps=4,
                                )
    return out

def call(kernel, load_dir, x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    kernel[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                # N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                load_dir=load_dir,
                                )
    return out


def main():
    ga_config = parse_args()

    random.seed(ga_config.seed)
    np.random.seed(ga_config.seed)
    torch.manual_seed(ga_config.seed)

    batch, heads, seq_len, dim = ga_config.Z, ga_config.H, ga_config.wl, ga_config.D_HEAD
    K=heads*dim

    embeddings_load = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="cuda")
    rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="cuda") * 0.2
    q_weights_load = torch.randn([heads * dim, heads * dim], dtype=torch.float16, device="cuda") * 0.2


    ga_config.total_flops = batch * seq_len * heads * dim
    ga_config.save_dir = f'{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}'

    if ga_config.load is None:
        load_dir = None
    elif ga_config.load == "auto":
        load_dir = f'{ga_config.default_out_path}/{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}'
    else:
        load_dir = ga_config.load


    @fgk_autotune(
        configs=[
		triton.Config({'N_SIZE': K, 'eps': 1e-6, 'BLOCK_N_SIZE':32}, num_stages=2, num_warps=4),

    ],
        key=['N_SIZE'],
        ret_ptr=2,
        ga_config=ga_config,
    )
    @jit
    def _ga(x_ptr, rms_w_ptr, output_ptr,
                    stride_x_batch, stride_x_m, stride_x_k,
                    stride_rms_w,
                    stride_out_batch, stride_out_m, stride_out_k,
                    N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
        pid_batch = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
        block_N = tl.arange(0, BLOCK_N_SIZE)
        var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
        for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
            offs_n = block_n_start_idx + block_N
            x_ptr_mask = offs_n < N_SIZE
            x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
            var += tl.math.pow(x.to(tl.float32), 2)

        var = tl.sum(var, axis=0) / N_SIZE
        rstd = tl.math.rsqrt(var + eps)

        # multiply by weight and add bias
        for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
            offs_n = block_n_start_idx + block_N
            x_ptr_mask = offs_n < N_SIZE
            rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

            x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
            x_hat = x * rstd
            out = x_hat * rms_w
            out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
            tl.store(output_ptr + out_off, out, mask=x_ptr_mask)

    call(_ga, load_dir, embeddings_load, rms_weights)

    if ga_config.tt:
        out_rms_triton = call_tt(x=embeddings_load, rms_w=rms_weights)

    if not ga_config.bench:
        return 
    torch.cuda.synchronize()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["NA"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            #x_vals=[2 ** i for i in range(8, 13)],  # Different possible values for `x_name`
            x_vals=[0],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=['ga', 'triton'],
            line_names=['ga', 'triton'],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name='bmm',
            args={"fp8_inputs": None},
        ))

    @triton.testing.perf_report(configs)
    def benchmark(NA, provider, fp8_inputs):
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'ga':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: call(_ga, load_dir, embeddings_load, rms_weights), quantiles=quantiles, warmup=100, rep=100)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: call_tt(embeddings_load, rms_weights), quantiles=quantiles, warmup=100, rep=100)
        perf = lambda ms: batch * heads * seq_len * dim * 1e-9 / (ms * 1e-3) # gflops
        return perf(ms), perf(max_ms), perf(min_ms)
    
    benchmark.run(show_plots=True, print_data=True)

if __name__ == '__main__':
    main()