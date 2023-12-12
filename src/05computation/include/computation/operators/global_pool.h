#ifndef COMPUTATION_GLOBAL_POOL_H
#define COMPUTATION_GLOBAL_POOL_H

#include "../operator.h"
#include "kernel/collectors/global_pool.h"

namespace refactor::computation {
    using kernel::PoolType;

    struct GlobalPool final : public Operator {
        PoolType type;

        constexpr GlobalPool(PoolType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(PoolType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GLOBAL_POOL_H
