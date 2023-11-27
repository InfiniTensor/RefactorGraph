#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"
#ifdef USE_CUDA
#include "functions.cuh"
#include "memory.cuh"
#endif

namespace refactor::hardware {

    Nvidia::Nvidia(std::string_view deviceTypeName,
                   int32_t typeId,
                   int32_t cardId)
        : Device(deviceTypeName,
                 typeId,
                 cardId,
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
        setDevice(cardId());
#endif
    }

    Arc<Device> Nvidia::build(
        std::string_view deviceTypeName,
        int32_t typeId,
        int32_t cardId,
        std::string_view args) {
        return Arc<Device>(new Nvidia(deviceTypeName, typeId, cardId));
    }

}// namespace refactor::hardware
