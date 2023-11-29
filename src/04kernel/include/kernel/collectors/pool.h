#ifndef KERNEL_POOL_H
#define KERNEL_POOL_H

#include "../attributes/pool_attributes.h"
#include "../collector.h"

namespace refactor::kernel {

    struct PoolCollector final : public InfoCollector {
        PoolType type;
        bool ceil;
        KernelShape kernelShape;
        PoolAttributes attributes;

        PoolCollector(decltype(_target), PoolType, bool, KernelShape, PoolAttributes) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_H
