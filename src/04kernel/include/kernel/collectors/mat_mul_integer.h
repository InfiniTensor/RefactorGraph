#ifndef KERNEL_MAT_MUL_INTEGER_H
#define KERNEL_MAT_MUL_INTEGER_H

#include "../collector.h"

namespace refactor::kernel {

    struct MatMulIntegerCollector final : public InfoCollector {

        constexpr MatMulIntegerCollector(decltype(_target) target) noexcept
            : InfoCollector(target) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_MAT_MUL_INTEGER_H
