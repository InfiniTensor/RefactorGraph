#ifndef COMPUTATION_POOL_H
#define COMPUTATION_POOL_H

#include "../operator.h"

namespace refactor::computation {

    enum class PoolType {
        Average,
        Lp,
        Max,
    };

    struct Pool final : public Operator {
        PoolType type;

        constexpr Pool(PoolType type_)
            : Operator(), type(type_) {}

        static size_t typeId(PoolType);
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_POOL_H
