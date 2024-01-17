#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"

#ifdef USE_CUDA
#include "memory.hh"
#include <cuda_runtime.h>

#define CUDA_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != cudaSuccess) {                             \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cudaGetErrorString(status), (int) status));        \
    }
#endif

namespace refactor::hardware {

    static Arc<Memory> cudaMemory(int32_t card) {
#ifdef USE_CUDA
        int deviceCount;
        CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));
        ASSERT(0 <= card && card < deviceCount, "Invalid card id: {}", card);
        CUDA_ASSERT(cudaSetDevice(card));

        size_t free, total;
        CUDA_ASSERT(cudaMemGetInfo(&free, &total));
        auto size = free * 9 / 10;
        cudaDeviceProp prop;
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
        size_t alignment = prop.textureAlignment;
        fmt::println("initializing Nvidia GPU {}, memory {} / {}, alloc {}, alignment {}",
                     card, free, total, size, alignment);
        return std::make_shared<MemPool>(
            std::make_shared<NvidiaMemory>(),
            size,
            alignment);
#else
        RUNTIME_ERROR("CUDA is not enabled");
#endif
    }

    Nvidia::Nvidia(int32_t card) : Device(card, cudaMemory(card)) {}

    void Nvidia::setContext() const {
#ifdef USE_CUDA
        CUDA_ASSERT(cudaSetDevice(_card));
#endif
    }

}// namespace refactor::hardware
