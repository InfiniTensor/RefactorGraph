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
    constexpr static DeviceKey CPU_KEY{{Device::Type::Cpu, 0}};

    Arc<Device> fetch(Device::Type type, int32_t card) {
        if (type == decltype(type)::Cpu) {
            auto it = DEVICES.find(CPU_KEY.i64);
            return it != DEVICES.end()
                       ? it->second
                       : DEVICES.emplace(CPU_KEY.i64, std::make_shared<Cpu>()).first->second;
        }
        auto it = DEVICES.find(DeviceKey{{type, card}}.i64);
        return it != DEVICES.end() ? it->second : nullptr;
    }
    Arc<Device> init(Device::Type type, int32_t card, std::string_view args) {
        if (type == decltype(type)::Cpu) {
            auto it = DEVICES.find(CPU_KEY.i64);
            return it != DEVICES.end()
                       ? it->second
                       : DEVICES.emplace(CPU_KEY.i64, std::make_shared<Cpu>()).first->second;
        }
        auto key = DeviceKey{{type, card}}.i64;
        if (auto it = DEVICES.find(key); it != DEVICES.end()) {
            return it->second;
        }
        auto device =
            type == Device::Type::Nvidia ? std::make_shared<Nvidia>(card)
                                         : UNREACHABLEX(Arc<Device>, "");
        return DEVICES.emplace(key, std::move(device)).first->second;
    }

}// namespace refactor::hardware::device
