#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"
#ifdef USE_CUDA
#include "functions.cuh"
#include "memory.cuh"
#endif

namespace refactor::hardware {

    static Arc<Memory> cudaMemory(int32_t card) {
#ifdef USE_CUDA
        ASSERT(0 <= card && card < getDeviceCount(), "Invalid card id: {}", card);
        setDevice(card);
        return std::make_shared<MemPool>(
            std::make_shared<NvidiaMemory>(),
            5ul << 30,
            256ul);
#else
        RUNTIME_ERROR("CUDA is not enabled");
#endif
    }

    Nvidia::Nvidia(int32_t card) : Device(card, cudaMemory(card)) {}

    void Nvidia::setContext() const noexcept {
#ifdef USE_CUDA
        setDevice(_card);
#endif
    }

}// namespace refactor::hardware
