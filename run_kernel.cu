#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA Error: " << errStr << std::endl; \
            exit(1); \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./run_kernel your_kernel.cubin" << std::endl;
        return 1;
    }

    const char* cubin_file = argv[1];

    // Init
    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    // Load cubin
    CUmodule mod;
    CHECK_CUDA(cuModuleLoad(&mod, cubin_file));

    // Get function (use your kernel name!)
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "mmleakyrelu"));
    //need to align with triton kernel name

    // Allocate input/output
    const int N = 1024;
    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_out, N * sizeof(float)));

    void* args[] = { &d_out, &d_in, &N };
    //need to align with triton kernel input args

    // Launch
    auto start = std::chrono::high_resolution_clock::now();
    //need to align with triton kernel config
    CHECK_CUDA(cuLaunchKernel(func,
        1, 1, 1,       // grid
        256, 1, 1,     // block
        0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Latency: " << latency_ms << " ms" << std::endl;

    // Cleanup
    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    return 0;
}
