#include "kernel/target.h"
#include "hardware/devices/cpu.h"
#include "hardware/devices/nvidia.h"
#include "hardware/mem_pool.h"

namespace refactor::kernel {
    using hardware::Device;

    Arc<Device> Target::device() const {
        switch (internal) {
            case Cpu: {
                Device::register_<hardware::Cpu>("cpu");
                return Device::init("cpu", 0, "");
            }
#ifdef USE_CUDA
            case NvidiaGpu: {
                Device::register_<hardware::Nvidia>("nvidia");
                return Device::init("nvidia", 0, "");
            }
#endif
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
