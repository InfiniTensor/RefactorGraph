#ifndef HARDWARE_DEVICE_MANAGER_H
#define HARDWARE_DEVICE_MANAGER_H

#include "device.h"

namespace refactor::hardware::device {

    Arc<Device> fetch(Device::Type, int32_t card);
    Arc<Device> init(Device::Type, int32_t card, std::string_view args);

}// namespace refactor::hardware::device

#endif// HARDWARE_DEVICE_MANAGER_H
