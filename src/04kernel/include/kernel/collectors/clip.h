#ifndef KERNEL_CLIP_H
#define KERNEL_CLIP_H

#include "../collector.h"

namespace refactor::kernel {

    struct ClipCollector final : public InfoCollector {

        explicit ClipCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_H
