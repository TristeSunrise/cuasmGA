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
from sass_kernel import SassKernel
from sassgen import extract_kernel_sass_from_bin, write_sass_file
from verify import test_via_cubin, gen_test_samples

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
    def test_performance():
        return 0

    sass, kernel_section = extract_kernel_sass_from_bin(bin)
    sasskernel = SassKernel(sass, kernel_section)
    pure_kernel_section = sasskernel._get_kernel()


    ga = GeneticAlgorithm(pure_kernel_section,test_correctness, test_performance)
    best = ga.run_ga(pure_kernel_section)

    sass = sasskernel._update_kernel(best.sass)
    cubin = write_sass_file(sass)
    asm['cubin'] = cubin
    


