#ifndef KERNEL_CONCAT_H
#define KERNEL_CONCAT_H

#include "../collector.h"

namespace refactor::kernel {

    struct ConcatCollector final : public InfoCollector {
        uint32_t axis;

        constexpr ConcatCollector(decltype(_target) target, uint32_t axis_) noexcept
            : InfoCollector(target), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONCAT_H
