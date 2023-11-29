#include "hardware/device.h"

namespace refactor::hardware {

    Device::Blob::Blob(decltype(_device) device, size_t size)
        : _device(device), _ptr(nullptr), _size(size) {
        _device->setContext();
        _ptr = _device->_mem->malloc(size);
    }

    Device::Blob::~Blob() {
        _device->setContext();
        _device->_mem->free(std::exchange(_ptr, nullptr));
    }
    void Device::Blob::copyFromHost(void const *ptr) const {
        _device->setContext();
        _device->_mem->copyHD(_ptr, ptr, _size);
    }
    void Device::Blob::copyFromHost(void const *ptr, size_t size) const {
        ASSERT(size <= _size, "size too large");
        _device->setContext();
        _device->_mem->copyHD(_ptr, ptr, size);
    }
    void Device::Blob::copyToHost(void *ptr) const {
        _device->setContext();
        _device->_mem->copyDH(ptr, _ptr, _size);
    }
    void Device::Blob::copyToHost(void *ptr, size_t size) const {
        ASSERT(size <= _size, "size too large");
        _device->setContext();
        _device->_mem->copyDH(ptr, _ptr, size);
    }
    void Device::Blob::copyFrom(Blob const &rhs) const {
        copyFrom(rhs, rhs._size);
    }
    void Device::Blob::copyFrom(Blob const &rhs, size_t size) const {
        ASSERT(size <= rhs._size && size <= _size, "size too large");
        if (_device->_mem == rhs._device->_mem) {
            _device->setContext();
            _device->_mem->copyDD(_ptr, rhs._ptr, size);
        } else if (rhs._device->type() == Device::Type::Cpu) {
            copyFromHost(rhs._ptr, size);
        } else {
            std::vector<uint8_t> tmp(size);
            rhs.copyToHost(tmp.data(), size);
            copyFromHost(tmp.data(), size);
        }
    }
    void Device::Blob::copyTo(Blob const &rhs) const {
        rhs.copyFrom(*this);
    }
    void Device::Blob::copyTo(Blob const &rhs, size_t size) const {
        rhs.copyFrom(*this, size);
    }

    Device::Device(decltype(_card) card, decltype(_mem) mem)
        : _card(card), _mem(std::move(mem)) {}

    void Device::setContext() const noexcept {}
    auto Device::malloc(size_t size) -> Arc<Blob> {
        return Arc<Blob>(new Blob(this, size));
    }
    auto Device::absorb(Arc<Blob> &&blob) -> Arc<Blob> {
        if (blob->_device == this) {
            return std::move(blob);
        }
        if (blob->_device->_mem == _mem) {
            blob->_device = this;
            return std::move(blob);
        }
        auto newBlob = malloc(blob->_size);
        newBlob->copyFrom(*blob);
        return newBlob;
    }

    static std::unordered_map<int64_t, Arc<Device>> DEVICES;

}// namespace refactor::hardware
