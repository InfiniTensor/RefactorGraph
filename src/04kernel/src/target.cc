#include "kernel/target.h"
#include "hardware/device_manager.h"
#include "hardware/mem_pool.h"

namespace refactor::kernel {
    using hardware::Device;

    Arc<Device> Target::device() const {
        switch (internal) {
            case Cpu:
                return hardware::device::init(Device::Type::Cpu, 0, "");
#ifdef USE_CUDA
            case NvidiaGpu:
                return hardware::device::init(Device::Type::Nvidia, 0, "");
#endif
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
