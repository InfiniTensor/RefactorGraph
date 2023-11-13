#ifndef KERNEL_SLICE_H
#define KERNEL_SLICE_H

#include "../attributes/slice_info.h"
#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct SliceCollector final : public InfoCollector {
        Target target;
        Dimensions dimentions;

        SliceCollector(Target, Dimensions) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SLICE_H
