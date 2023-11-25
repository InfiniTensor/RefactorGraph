#include "hardware/device.h"
#include <unordered_map>

namespace refactor::hardware {

    Device::Device(decltype(_deviceTypeName) deviceTypeName,
                   decltype(_typeId) typeId,
                   decltype(_cardId) cardId,
                   decltype(_mem) mem)
        : _deviceTypeName(deviceTypeName),
          _typeId(typeId),
          _cardId(cardId),
          _mem(std::move(mem)) {}

    void *Device::malloc(size_t size) {
        return _mem->malloc(size);
    }
    void Device::free(void *ptr) {
        _mem->free(ptr);
    }
    void *Device::copyHD(void *dst, void const *src, size_t bytes) const {
        return _mem->copyHD(dst, src, bytes);
    }
    void *Device::copyDH(void *dst, void const *src, size_t bytes) const {
        return _mem->copyDH(dst, src, bytes);
    }
    void *Device::copyDD(void *dst, void const *src, size_t bytes) const {
        return _mem->copyDD(dst, src, bytes);
    }

    struct DeviceType {
        int32_t id;
        Device::Builder builder;
    };

    static std::unordered_map<std::string, DeviceType> BUILDERS;
    static std::unordered_map<int64_t, Arc<Device>> DEVICES;

    void Device::register_(std::string name, Builder builder) {
        auto [it, ok] = BUILDERS.try_emplace(
            std::move(name),
            DeviceType{
                .id = static_cast<int32_t>(BUILDERS.size()),
                .builder = builder,
            });
        ASSERT(ok || it->second.builder == builder, "Duplicate device type");
    }

    Arc<Device> Device::init(
        std::string const &deviceName,
        decltype(_cardId) cardId,
        std::string_view args) {
        if (auto it = BUILDERS.find(deviceName); it != BUILDERS.end()) {
            auto [typeId, builder] = it->second;
            union {
                int32_t i32[2];
                int64_t i64;
            } id{.i32 = {typeId, cardId}};

            auto it_ = DEVICES.find(id.i64);
            if (it_ == DEVICES.end()) {
                it_ = DEVICES.emplace(id.i64, builder(it->first, typeId, cardId, args)).first;
            }
            return it_->second;
        }
        UNREACHABLEX(void, "Unknown type \"{}\"", deviceName);
    }

}// namespace refactor::hardware
