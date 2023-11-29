#ifndef KERNEL_WHERE_H
#define KERNEL_WHERE_H

#include "../collector.h"

namespace refactor::kernel {

    struct WhereCollector final : public InfoCollector {
        constexpr WhereCollector(decltype(_target) target) noexcept
            : InfoCollector(target) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_WHERE_H
