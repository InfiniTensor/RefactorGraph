#ifndef HARDWARE_DEVICES_NVIDIA_H
#define HARDWARE_DEVICES_NVIDIA_H

#include "../device.h"

#define CUDA_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != cudaSuccess) {                             \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cudaGetErrorString(status), (int) status));        \
    }

namespace refactor::hardware {

    class Nvidia final : public Device {
    public:
        explicit Nvidia(int32_t card);
        void setContext() const final;
        Type type() const noexcept final {
            return Type::Nvidia;
        }
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_H
