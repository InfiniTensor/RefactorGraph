#ifndef HARDWARE_DEVICES_CPU_H
#define HARDWARE_DEVICES_CPU_H

#include "../device.h"

namespace refactor::hardware {

    class Cpu final : public Device {
    public:
        Cpu();

        Type type() const noexcept final {
            return Type::Cpu;
        }
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_CPU_H
