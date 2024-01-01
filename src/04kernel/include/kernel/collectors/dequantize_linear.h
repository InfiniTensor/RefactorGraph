#ifndef KERNEL_DEQUANTIZE_LINEAR_H
#define KERNEL_DEQUANTIZE_LINEAR_H

#include "../collector.h"

namespace refactor::kernel {

    struct DequantizeLinearCollector final : public InfoCollector {

        explicit DequantizeLinearCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_DEQUANTIZE_LINEAR_H
