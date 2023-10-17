#ifndef KERNEL_POOL_H
#define KERNEL_POOL_H

#include "../collector.h"
#include "../target.h"
#include "pool_attributes.hh"

namespace refactor::kernel {

    enum class PoolType {
        Average,
        Lp,
        Max,
    };

    struct PoolCollector final : public InfoCollector {
        Target target;
        PoolType type;
        PoolAttributes attributes;

        PoolCollector(Target, PoolType, PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_H
