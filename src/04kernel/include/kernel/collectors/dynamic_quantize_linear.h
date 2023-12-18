#ifndef KERNEL_DYNAMIC_QUANTIZE_LINEAR_H
#define KERNEL_DYNAMIC_QUANTIZE_LINEAR_H

#include "../collector.h"

namespace refactor::kernel {

    struct DynamicQuantizeLinearCollector final : public InfoCollector {

        explicit DynamicQuantizeLinearCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_DYNAMIC_QUANTIZE_LINEAR_H
