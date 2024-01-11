#ifndef KERNEL_SIMPLE_UNARY_H
#define KERNEL_SIMPLE_UNARY_H

#include "../collector.h"

namespace refactor::kernel {

    enum class SimpleUnaryType : uint8_t {
        Abs,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
        Tanh,
        Relu,
        Sqrt,
        Sigmoid,
        Erf,
        Neg,
        Not,
        HardSwish,
    };

    std::string_view unaryName(SimpleUnaryType type);

    struct SimpleUnaryCollector final : public InfoCollector {
        SimpleUnaryType type;

        constexpr explicit SimpleUnaryCollector(decltype(_target) target, SimpleUnaryType type_) noexcept
            : InfoCollector(target), type(type_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_H
