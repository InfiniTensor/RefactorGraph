#include "hardware/device_manager.h"
#include "hardware/devices/cpu.h"
#include "hardware/devices/mlu.h"
#include "hardware/devices/nvidia.h"

namespace refactor::hardware::device {

    static Arc<Device> cpu() {
        static auto cpu = std::make_shared<Cpu>();
        return cpu;
    }
    static std::unordered_map<int32_t, std::unordered_map<int32_t, Arc<Device>>> DEVICES;
    constexpr static auto CPU_KEY = static_cast<int32_t>(Device::Type::Cpu);

    Arc<Device> fetch(Device::Type type) {
        auto type_ = static_cast<int32_t>(type);
        if (type_ == CPU_KEY) { return cpu(); }
        if (auto kind = DEVICES.find(type_); kind != DEVICES.end()) {
            if (auto it = kind->second.begin(); it != kind->second.end()) {
                return it->second;
            }
        }
        return init(type, 0, "");
    }
    Arc<Device> fetch(Device::Type type, int32_t card) {
        auto type_ = static_cast<int32_t>(type);
        if (type_ == CPU_KEY) { return cpu(); }
        if (auto kind = DEVICES.find(type_); kind != DEVICES.end()) {
            if (auto it = kind->second.find(card); it != kind->second.end()) {
                return it->second;
            }
        }
        return nullptr;
    }
    Arc<Device> init(Device::Type type, int32_t card, std::string_view args) {
        if (auto device = fetch(type, card); device) { return device; }

        using T = Device::Type;
        // clang-format off
        auto device = type == T::Nvidia ? std::make_shared<Nvidia>(card)
                    : type == T::Mlu    ? std::make_shared<Mlu>(card)
                    : UNREACHABLEX(Arc<Device>, "");
        // clang-format on
        auto [kind, ok] = DEVICES.try_emplace(static_cast<int32_t>(type));
        return kind->second.emplace(card, std::move(device)).first->second;
    }

}// namespace refactor::hardware::device
