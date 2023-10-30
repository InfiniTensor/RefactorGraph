#ifndef KERNEL_MAT_MUL_H
#define KERNEL_MAT_MUL_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct MatMulCollector final : public InfoCollector {
        Target target;

        constexpr MatMulCollector(Target target_) noexcept
            : InfoCollector(), target(target_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_MAT_MUL_H
