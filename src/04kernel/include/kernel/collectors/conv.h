#ifndef KERNEL_CONV_H
#define KERNEL_CONV_H

#include "../collector.h"
#include "../target.h"
#include "pool_attributes.hh"

namespace refactor::kernel {

    struct ConvCollector final : public InfoCollector {
        Target target;
        PoolAttributes poolAttributes;

        ConvCollector(Target, PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_H
