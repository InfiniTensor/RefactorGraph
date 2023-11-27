#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"
#ifdef USE_CUDA
#include "functions.cuh"
#include "memory.cuh"
#endif

namespace refactor::hardware {

    Nvidia::Nvidia(int32_t card)
        : Device(card,
#ifdef USE_CUDA
                 std::make_shared<MemPool>(
                     std::make_shared<NvidiaMemory>(),
                     5ul << 30,
                     256ul)
#else
                 nullptr
#endif
          ) {
    }

    void Nvidia::setContext() const noexcept {
#ifdef USE_CUDA
        setDevice(_card);
#endif
    }

}// namespace refactor::hardware
