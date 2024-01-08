#ifndef HARDWARE_DEVICE_H
#define HARDWARE_DEVICE_H

#include "common.h"
#include "memory.h"

namespace refactor::hardware {

    class Device {
    public:
        enum class Type : int32_t {
            Cpu,
            Nvidia,
        };

    protected:
        int32_t _card;
        Arc<Memory> _mem;
        Device(decltype(_card), decltype(_mem));

    public:
        class Blob {
            friend class Device;

            Device *_device;
            void *_ptr;
            size_t _size;

            Blob(decltype(_device) device, size_t);

        public:
            ~Blob();
            // clang-format off
            void copyFromHost(void const *        ) const;
            void copyFromHost(void const *, size_t) const;
            void copyToHost  (void       *        ) const;
            void copyToHost  (void       *, size_t) const;
            void copyFrom    (Blob const &        ) const;
            void copyFrom    (Blob const &, size_t) const;
            void copyTo      (Blob const &        ) const;
            void copyTo      (Blob const &, size_t) const;
            // clang-format on
            constexpr size_t size() const noexcept { return _size; }
            constexpr void *get() const noexcept { return _ptr; }
            template<class T> constexpr T *get() const noexcept { return static_cast<T *>(_ptr); }
            constexpr operator void *() const noexcept { return get(); }
            constexpr operator bool() const noexcept { return get(); }
        };
        friend class Blob;

        virtual ~Device() = default;
        virtual Type type() const noexcept = 0;
        virtual void setContext() const;

        Arc<Blob> malloc(size_t);
        Arc<Blob> absorb(Arc<Blob> &&);
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICE_H
