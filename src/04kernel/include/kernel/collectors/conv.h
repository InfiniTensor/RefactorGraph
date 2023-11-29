#ifndef KERNEL_CONV_H
#define KERNEL_CONV_H

#include "../attributes/pool_attributes.h"
#include "../collector.h"

namespace refactor::kernel {

    struct ConvCollector final : public InfoCollector {
        PoolAttributes poolAttrs;

        ConvCollector(decltype(_target), PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_H
