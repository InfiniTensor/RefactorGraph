﻿#include "functions.cuh"
#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"
#include "memory.cuh"

namespace refactor::hardware {

    static Arc<Memory> cudaMemory(int32_t card) {
#ifdef USE_CUDA
        ASSERT(0 <= card && card < getDeviceCount(), "Invalid card id: {}", card);
        setDevice(card);
        auto [free, total] = getMemInfo();
        auto size = std::min(free, std::max(5ul << 30, total * 4 / 5));
        fmt::println("initializing Nvidia GPU {}, memory {} / {}, alloc {}",
                     card, free, total, size);
        return std::make_shared<MemPool>(
            std::make_shared<NvidiaMemory>(),
            size,
            256ul);
#else
        return nullptr;
#endif
    }

    Nvidia::Nvidia(int32_t card) : Device(card, cudaMemory(card)) {}

    void Nvidia::setContext() const noexcept {
        setDevice(_card);
    }

}// namespace refactor::hardware
