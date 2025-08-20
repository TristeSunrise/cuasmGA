import os
import time
import pickle
import tempfile

import torch

from CuAsm.CubinFile import CubinFile
from CuAsm.CuAsmParser import CuAsmParser

from compiler import CompiledKernel as fgk_CompiledKernel
from verify import test_via_cubin
from logger import get_logger

logger = get_logger(__name__)


# YAPF: disable
def run_selection(
    so_path, metadata, asm,

    # args
    args, sig_key, non_constexpr_arg_values,
    ret_ptr, test_inputs, test_outputs,

    # kernel args
    grid_0, grid_1, grid_2, stream,

    enter_hook, exit_hook,

    cubin_dir_path, n_test_samples,
):

    t1 = time.perf_counter()
    
    rankings = {}
    if cubin_dir_path.endswith('.pkl'):
        # specified a record optimized by fgk
        with open(cubin_dir_path, 'rb') as f:
            data = pickle.load(f)
        rankings[0] = data
    else:
        # select a list of record in the search result directory
        for i, fn in enumerate(os.listdir(cubin_dir_path)):
            if fn == 'cache_config.pkl':
                continue
            if not fn.endswith('.pkl'):
                continue

            with open(os.path.join(cubin_dir_path, fn), 'rb') as f:
                data = pickle.load(f)

            rankings[fn] = data

    if len(rankings) == 0:
        raise RuntimeError(f'no valid cubin found in {cubin_dir_path}')

    logger.info(f'found {len(rankings)} cubins in {cubin_dir_path}')
    # for run, data in sorted(rankings.items(),
    #                         key=lambda x: x[1]['final_perf'],
    #                         reverse=True):
    #     print(f'run {run}; perf: {data["final_perf"]:.2f}:{data["init_perf"]:.2f}')
    test_all = False
    if os.getenv("SIP_TESTALL", "0") == "1":
        test_all = True
    test_batch_size = int(os.getenv("SIP_TESTBATCH", "1"))

    # run verificaiton greedily
    cnt = 0
    for run, data in sorted(rankings.items(),
                            key=lambda x: x[1]['final_perf'],
                            reverse=True):
        cubin = data['cubin']
        try:
            ok = test_via_cubin(
                so_path,
                metadata,
                asm,

                # args
                args,
                sig_key,
                non_constexpr_arg_values,
                ret_ptr,
                None,  # static test samples

                # kernel args
                grid_0,
                grid_1,
                grid_2,
                stream,

                # enter_hook, exit_hook,
                None,
                None,
                cubin,
                n_test_samples,
                test_batch_size,
            )
        except Exception as e:
            logger.warning(f'run {run} verify failed: {e}')
            ok = False

        torch.cuda.empty_cache()  # free test memory
        if ok:
            cnt += 1
            logger.info(f'run {run} verified ok')
            if not test_all:
                break
        else:
            logger.warning(f'run {run} verified failed')

    t2 = time.perf_counter()
    logger.info(f'verification time: {t2 - t1:.2f}s')

    if test_all:
        logger.info(f'verified {cnt}/{len(rankings)} kernels')


    if not ok:
        raise RuntimeError(f'verification failed for kernel in {cubin_dir_path}')

    opt_asm = {
        'cubin': cubin,
    }
    opt_bin = fgk_CompiledKernel(so_path, metadata, opt_asm)
    return opt_bin
