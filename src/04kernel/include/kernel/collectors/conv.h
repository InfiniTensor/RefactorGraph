#ifndef KERNEL_CONV_H
#define KERNEL_CONV_H

#include "../attributes/pool_attributes.h"
#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct ConvCollector final : public InfoCollector {
        Target target;
        PoolAttributes poolAttrs;

        ConvCollector(Target, PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_H
