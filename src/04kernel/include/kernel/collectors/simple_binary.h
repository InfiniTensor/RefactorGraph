#ifndef KERNEL_SIMPLE_BINARY_H
#define KERNEL_SIMPLE_BINARY_H

#include "../collector.h"
#include "../target.h"

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
        Target target;

        constexpr SimpleBinaryCollector(Target target_, SimpleBinaryType type_) noexcept
            : InfoCollector(), target(target_), type(type_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_BINARY_H
