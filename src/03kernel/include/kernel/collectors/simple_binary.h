#ifndef KERNEL_SIMPLE_BINARY_H
#define KERNEL_SIMPLE_BINARY_H

#include "../collector.h"

namespace refactor::kernel {

    enum class SimpleBinaryType {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        And,
        Or,
        Xor,
    };

    struct SimpleBinaryCollector final : public InfoCollector {
        SimpleBinaryType type;

        constexpr explicit SimpleBinaryCollector(SimpleBinaryType type_) noexcept
            : InfoCollector(), type(type_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_BINARY_H
