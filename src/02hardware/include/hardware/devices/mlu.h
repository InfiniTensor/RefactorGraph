#ifndef HARDWARE_DEVICES_MLU_H
#define HARDWARE_DEVICES_MLU_H

#include "../device.h"

namespace refactor::hardware {

    class Mlu final : public Device {
    public:
        explicit Mlu(int32_t card);
        void setContext() const noexcept final;
        Type type() const noexcept final {
            return Type::Mlu;
        }
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_MLU_H
