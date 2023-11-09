#include "kernel/cuda/threads_distributer.cuh"
#include "macro.cuh"

namespace refactor::kernel::cuda {

    // constexpr static unsigned long long
    //     THREADS_PER_BLOCK_MAX           = 1024 ,
    //     THREADS_PER_WRAP                = 32   ,
    //     THREADS_PER_SM_MAX              = 2048 ,
    //     BLOCKS_PER_SM_MAX               = 16   ,
    //     THREADS_PER_BLOCK_MIN           = 128  , // THREADS_PER_SM_MAX / BLOCKS_PER_SM_MAX
    //     REGISTERS_PER_BLOCK             = 65536,
    //     REGISTERS_PER_THREAD            = 255  ,
    //     THREADS_PER_BLOCK_MAX_REGISTERS = 256  ; // REGISTERS_PER_BLOCK / REGISTERS_PER_THREAD

    // ...so
    constexpr static int BlockSize = 256, MaxWaves = 32;

    ThreadsDistributer::ThreadsDistributer()
        : _maxGridSize(0) {
        int dev, mpc, tps;
        CUDA_ASSERT(cudaGetDevice(&dev));
        CUDA_ASSERT(cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, dev));
        CUDA_ASSERT(cudaDeviceGetAttribute(&tps, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
        _maxGridSize = mpc * tps * MaxWaves / BlockSize;
    }

    auto ThreadsDistributer::operator()(size_t n) const noexcept -> KernelLaunchParameters {
        auto gridSize = (n + BlockSize - 1) / BlockSize;
        ASSERT(gridSize <= std::numeric_limits<int>::max(), "");
        return {std::clamp(static_cast<int>(gridSize), 1, _maxGridSize), BlockSize, n};
    }

}// namespace refactor::kernel::cuda
