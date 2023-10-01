#ifndef COMPUTATION_GLOBAL_POOL_H
#define COMPUTATION_GLOBAL_POOL_H

#include "../operator.h"
#include "pool.h"

namespace refactor::computation {

    struct GlobalPool final : public Operator {
        PoolType type;

        constexpr GlobalPool(PoolType type_)
            : Operator(), type(type_) {}

        static size_t typeId(PoolType);
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GLOBAL_POOL_H
