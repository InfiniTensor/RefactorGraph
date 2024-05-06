#ifndef KERNEL_TOPK_H
#define KERNEL_TOPK_H

#include "../collector.h"

namespace refactor::kernel {

    struct TopKCollector final : public InfoCollector {
        uint32_t topk, axis;

        constexpr TopKCollector(decltype(_target) target, uint32_t topk, uint32_t axis_) noexcept
            : InfoCollector(target), topk(topk), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_H
