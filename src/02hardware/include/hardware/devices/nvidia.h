#ifndef HARDWARE_DEVICES_NVIDIA_H
#define HARDWARE_DEVICES_NVIDIA_H

#include "../device.h"

namespace refactor::hardware {

    class Nvidia final : public Device {
    public:
        explicit Nvidia(int32_t card);
        void setContext() const noexcept final;
        constexpr Type type() const noexcept final {
            return Type::Nvidia;
        }
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_H
