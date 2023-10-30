#ifndef KERNEL_WHERE_H
#define KERNEL_WHERE_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct WhereCollector final : public InfoCollector {
        Target target;

        constexpr WhereCollector(Target target_) noexcept
            : InfoCollector(), target(target_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_WHERE_H
