#ifndef HARDWARE_DEVICES_NVIDIA_H
#define HARDWARE_DEVICES_NVIDIA_H

#include "../device.h"

namespace refactor::hardware {

    class Nvidia final : public Device {
        Nvidia(std::string_view deviceTypeName,
               int32_t typeId,
               int32_t cardId);

    public:
        void setContext() const noexcept final;

        static Arc<Device> build(
            std::string_view deviceTypeName,
            int32_t typeId,
            int32_t cardId,
            std::string_view args);
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_H
