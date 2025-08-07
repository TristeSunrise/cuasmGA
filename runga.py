import os
import math
import random
import tempfile
import time
from functools import partial
from copy import deepcopy

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

from ga import GeneticAlgorithm

# mutation
from logger import get_logger
# from cuasmrl.utils.record import save_data, read_data
from record import save_data
from sass_kernel import SassKernel
from sassgen import extract_kernel_sass_from_bin, write_sass_file
from verify import test_via_cubin, gen_test_samples
from triton.testing import do_bench

logger = get_logger(__name__)


def run_ga(
    # kernel
    bin,
    so_path,
    metadata,
    asm,
    ret_ptr,
    args,
    sig_key,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,
    # ga config
    #TODO: add ga config
    config,
):


    # TODO gen static test samples
    test_correctness = partial(
        test_via_cubin,
        so_path,
        metadata,
        asm,
        args,
        sig_key,
        non_constexpr_arg_values,
        ret_ptr,
        None,

        #
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
    )

    #TODO: test performance
    def test_performance(individual):
        assemble_ok = True
        cubin = None
        try:
            sass = sasskernel._update_kernel(individual.sass)
            cubin = write_sass_file(sass)
            bin.asm['cubin'] = cubin
        except Exception as e:
            print(f'Assemble failed: {e}')
            assemble_ok = False
            cubin = None

            
        # BENCH
        fn = lambda: bin.c_wrapper(
            grid_0,
            grid_1,
            grid_2,
            bin.num_warps,
            bin.num_ctas,
            bin.clusterDims[0],
            bin.clusterDims[1],
            bin.clusterDims[2],
            bin.shared,
            stream,
            bin.cu_function,
            launch_enter_hook,
            launch_exit_hook,
            bin,
            *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
        )
        if assemble_ok:
            try:
                ms = triton.testing.do_bench(fn, warmup=100, rep=100)
                # ms = do_bench(fn, 100, 100)
            except RuntimeError as run_err:
                # likely a cuda error
                logger.error(f'CUDA? Runtime Err: {run_err}')
                ms = -1
                cubin = None
            except Exception as e:
                logger.error(f'Other error: {e}')
                raise e
        else:
            ms = -1

        if config.total_flops is not None:
            tflops = config.total_flops / ms * 1e-9
            # print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops, cubin

        # print(f'ms: {ms:.3f};')
        raise NotImplementedError()



    sass, kernel_section = extract_kernel_sass_from_bin(bin)
    sasskernel = SassKernel(sass, kernel_section)
    pure_kernel_section = sasskernel._get_kernel()


    ga = GeneticAlgorithm(pure_kernel_section,test_correctness, test_performance)
    best = ga.run_ga(pure_kernel_section)

    sass = sasskernel._update_kernel(best.sass)
    cubin = write_sass_file(sass)
    asm['cubin'] = cubin
    bin.asm['cubin'] = cubin

    save_path = os.path.join(config.default_out_path, config.save_dir)
    save_data(bin, best.fitness, save_path)
    


