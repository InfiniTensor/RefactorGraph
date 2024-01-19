#ifndef KERNEL_PAD_H
#define KERNEL_PAD_H

#include "../attributes/pad_info.h"
#include "../collector.h"

namespace refactor::kernel {

    struct PadCollector final : public InfoCollector {
        PadsShape pads;
        PadType mode;

        explicit PadCollector(decltype(_target) target, PadsShape const &pads_, PadType mode_) noexcept
            : InfoCollector(target), pads(std::move(pads_)), mode(mode_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif// KERNEL_PAD_H
