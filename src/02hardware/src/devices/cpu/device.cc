#include "hardware/devices/cpu.h"
#include "hardware/mem_pool.h"
#include "memory.hh"

namespace refactor::hardware {

    Cpu::Cpu(std::string_view deviceTypeName,
             int32_t typeId,
             int32_t cardId)
        : Device(deviceTypeName,
                 typeId,
                 cardId,
                 std::make_shared<CpuMemory>()) {}

    Arc<Device> Cpu::build(
        std::string_view deviceTypeName,
        int32_t typeId,
        int32_t cardId,
        std::string_view args) {
        return Arc<Device>(new Cpu(deviceTypeName, typeId, cardId));
    }

}// namespace refactor::hardware
