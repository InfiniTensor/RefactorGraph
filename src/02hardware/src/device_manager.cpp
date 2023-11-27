#include "hardware/device_manager.h"
#include "hardware/devices/cpu.h"
#include "hardware/devices/nvidia.h"
#include <unordered_map>

namespace refactor::hardware::device {

    static std::unordered_map<int64_t, Arc<Device>> DEVICES;
    union DeviceKey {
        struct {
            Device::Type type;
            int32_t card;
        };
        int64_t i64;
    };

    Arc<Device> fetch(Device::Type type, int32_t card) {
        auto it = DEVICES.find(DeviceKey{{type, card}}.i64);
        return it != DEVICES.end()         ? it->second
               : type == Device::Type::Cpu ? std::make_shared<Cpu>()
                                           : nullptr;
    }
    Arc<Device> init(Device::Type type, int32_t card, std::string_view args) {
        auto key = DeviceKey{{type, card}}.i64;
        if (auto it = DEVICES.find(key); it != DEVICES.end()) {
            return it->second;
        }
        auto device = type == Device::Type::Cpu      ? std::make_shared<Cpu>()
                      : type == Device::Type::Nvidia ? std::make_shared<Nvidia>(card)
                                                     : UNREACHABLEX(Arc<Device>, "");
        return DEVICES.emplace(key, std::move(device)).first->second;
    }

}// namespace refactor::hardware::device
