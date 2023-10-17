#ifndef KERNEL_POOL_H
#define KERNEL_POOL_H

#include "../collector.h"
#include "../target.h"
#include "pool_attributes.hh"

namespace refactor::kernel {

    struct PoolCollector final : public InfoCollector {
        Target target;
        PoolType type;
        bool ceil;
        absl::InlinedVector<uint16_t, 2> kernelShape;
        PoolAttributes attributes;

        PoolCollector(Target, PoolType, bool, decltype(kernelShape), PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_H
