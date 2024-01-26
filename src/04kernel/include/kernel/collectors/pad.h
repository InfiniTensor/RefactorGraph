#ifndef KERNEL_PAD_H
#define KERNEL_PAD_H

#include "../attributes/pad_info.h"
#include "../collector.h"

namespace refactor::kernel {

    struct PadCollector final : public InfoCollector {
        PadDimension dims;
        PadType mode;

        explicit PadCollector(decltype(_target) target, PadDimension const &dims_, PadType mode_) noexcept
            : InfoCollector(target), dims(std::move(dims_)), mode(mode_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif// KERNEL_PAD_H
