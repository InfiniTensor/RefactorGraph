#include "hardware/device.h"
#include <unordered_map>

namespace refactor::hardware {

    Device::Blob::Blob(decltype(_device) device, size_t size)
        : _device(device), _ptr(nullptr) {
        _device.setContext();
        _ptr = _device._mem->malloc(size);
    }

    Device::Blob::~Blob() {
        _device.setContext();
        _device._mem->free(std::exchange(_ptr, nullptr));
    }
    void Device::Blob::copyFromHost(void const *ptr, size_t size) const {
        _device.setContext();
        _device._mem->copyHD(_ptr, ptr, size);
    }
    void Device::Blob::copyToHost(void *ptr, size_t size) const {
        _device.setContext();
        _device._mem->copyDH(ptr, _ptr, size);
    }
    void Device::Blob::copyFrom(Blob const &rhs, size_t size) const {
        _device.setContext();
        if (_device._mem == rhs._device._mem) {
            _device._mem->copyDD(_ptr, rhs._ptr, size);
        } else {
            std::vector<uint8_t> tmp(size);
            rhs.copyToHost(tmp.data(), size);
            copyFromHost(tmp.data(), size);
        }
    }
    void Device::Blob::copyTo(Blob const &rhs, size_t size) const {
        rhs.copyFrom(*this, size);
    }

    Device::Device(decltype(_deviceTypeName) deviceTypeName,
                   decltype(_typeId) typeId,
                   decltype(_cardId) cardId,
                   decltype(_mem) mem)
        : _deviceTypeName(deviceTypeName),
          _typeId(typeId),
          _cardId(cardId),
          _mem(std::move(mem)) {}

    void Device::setContext() const noexcept {}
    auto Device::malloc(size_t size) -> Arc<Blob> {
        return Arc<Blob>(new Blob(*this, size));
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
