#ifndef HARDWARE_DEVICES_CPU_H
#define HARDWARE_DEVICES_CPU_H

#include "../device.h"

namespace refactor::hardware {

    class Cpu final : public Device {
        Cpu(std::string_view deviceTypeName,
            int32_t typeId,
            int32_t cardId);

    public:
        static Arc<Device> build(
            std::string_view deviceTypeName,
            int32_t typeId,
            int32_t cardId,
            std::string_view args);
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_CPU_H
