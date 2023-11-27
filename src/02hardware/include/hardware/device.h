#ifndef HARDWARE_DEVICE_H
#define HARDWARE_DEVICE_H

#include "common.h"
#include "memory.h"

namespace refactor::hardware {

    class Device {
        std::string_view _deviceTypeName;
        int32_t _typeId, _cardId;
        Arc<Memory> _mem;

    protected:
        Device(decltype(_deviceTypeName),
               decltype(_typeId),
               decltype(_cardId),
               decltype(_mem));

    public:
        virtual ~Device() = default;

        constexpr int32_t typeId() const noexcept { return _typeId; }
        constexpr int32_t cardId() const noexcept { return _cardId; }
        constexpr std::string_view deviceTypeName() const noexcept { return _deviceTypeName; }

        void *malloc(size_t);
        void free(void *);
        void *copyHD(void *dst, void const *src, size_t bytes) const;
        void *copyDH(void *dst, void const *src, size_t bytes) const;
        void *copyDD(void *dst, void const *src, size_t bytes) const;

        using Builder = Arc<Device> (*)(decltype(_deviceTypeName),
                                        decltype(_typeId),
                                        decltype(_cardId),
                                        std::string_view);
        static void register_(std::string, Builder);
        template<class T> static void register_(std::string deviceName) {
            register_(std::move(deviceName), T::build);
        }

        static Arc<Device> init(std::string const &deviceTypeName, decltype(_cardId), std::string_view);
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICE_H
