#ifndef KERNEL_GLOBAL_POOL_H
#define KERNEL_GLOBAL_POOL_H

#include "../attributes/pool_attributes.h"
#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct GlobalPoolCollector final : public InfoCollector {
        Target target;
        PoolType type;

        GlobalPoolCollector(Target, PoolType) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_GLOBAL_POOL_H
