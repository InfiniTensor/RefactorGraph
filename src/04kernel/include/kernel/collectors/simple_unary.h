#ifndef KERNEL_SIMPLE_UNARY_H
#define KERNEL_SIMPLE_UNARY_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    enum class SimpleUnaryType {
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
        Not,
    };

    struct SimpleUnaryCollector final : public InfoCollector {
        SimpleUnaryType type;
        Target target;

        constexpr explicit SimpleUnaryCollector(SimpleUnaryType type_, Target target_) noexcept
            : InfoCollector(), type(type_), target(target_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_H
