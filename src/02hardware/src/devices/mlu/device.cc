#include "functions.cc"
#include "hardware/devices/mlu.h"
#include "hardware/mem_pool.h"
#include "memory.hh"

namespace refactor::hardware {

    static Arc<Memory> bangMemory(int32_t card) {
        ASSERT(0 <= card && card < getDeviceCount(), "Invalid card id: {}", card);
        setDevice(card);
        auto [free, total] = getMemInfo();
        auto size = std::min(free, std::max(5ul << 30, total * 4 / 5));
        fmt::println("initializing Nvidia GPU {}, memory {} / {}, alloc {}",
                     card, free, total, size);
        return std::make_shared<MemPool>(
            std::make_shared<MluMemory>(),
            size,
            256ul);
    }

    Mlu::Mlu(int32_t card) : Device(card, bangMemory(card)) {}

    void Mlu::setContext() const noexcept {
        setDevice(_card);
    }

}// namespace refactor::hardware
