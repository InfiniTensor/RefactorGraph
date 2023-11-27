#include "hardware/devices/cpu.h"
#include "hardware/mem_pool.h"
#include "memory.hh"

namespace refactor::hardware {

    static Arc<Memory> cpuMemory() {
        static auto instance = std::make_shared<CpuMemory>();
        return instance;
    }

    Cpu::Cpu() : Device(0, cpuMemory()) {}

}// namespace refactor::hardware
