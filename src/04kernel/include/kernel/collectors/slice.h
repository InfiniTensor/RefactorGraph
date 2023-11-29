#ifndef KERNEL_SLICE_H
#define KERNEL_SLICE_H

#include "../attributes/slice_info.h"
#include "../collector.h"

namespace refactor::kernel {

    struct SliceCollector final : public InfoCollector {
        Dimensions dimentions;

        SliceCollector(decltype(_target), Dimensions) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SLICE_H
