#ifndef KERNEL_MOE_H
#define KERNEL_MOE_H

#include "../collector.h"

namespace refactor::kernel {

    struct AssignPosCollector final : public InfoCollector {
        uint32_t topk,numExperts;
        constexpr AssignPosCollector(decltype(_target) target, uint32_t topk, uint32_t numExperts) noexcept
            : InfoCollector(target) ,topk(topk), numExperts(numExperts){}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

    struct ReorderCollector final : public InfoCollector {
        bool scatter;
        uint32_t topk;
        constexpr ReorderCollector(decltype(_target) target, bool scatter, uint32_t topk) noexcept
            : InfoCollector(target) ,scatter(scatter), topk(topk){}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_H
