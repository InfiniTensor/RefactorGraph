#ifndef KERNEL_SIMPLE_BINARY_H
#define KERNEL_SIMPLE_BINARY_H

#include "../collector.h"

namespace refactor::kernel {

    enum class SimpleBinaryType : uint8_t {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        And,
        Or,
        Xor,
        Mod,
        Fmod,
    };

    std::string_view opName(SimpleBinaryType type);

    struct SimpleBinaryCollector final : public InfoCollector {
        SimpleBinaryType type;

        constexpr SimpleBinaryCollector(decltype(_target) target, SimpleBinaryType type_) noexcept
            : InfoCollector(target), type(type_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_BINARY_H
