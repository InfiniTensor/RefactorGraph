#include "hardware/devices/cpu.h"
#include "hardware/mem_pool.h"
#include "memory.hh"

namespace refactor::hardware {

    Cpu::Cpu() : Device(0, std::make_shared<CpuMemory>()) {}

}// namespace refactor::hardware
