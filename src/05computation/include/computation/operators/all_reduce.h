#ifndef COMPUTATION_ALL_REDUCE_H
#define COMPUTATION_ALL_REDUCE_H

#include "../operator.h"
#include "kernel/attributes/communication.h"

namespace refactor::computation {

    struct AllReduce final : public Operator {
        kernel::AllReduceType type;

        constexpr explicit AllReduce(kernel::AllReduceType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(kernel::AllReduceType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_ALL_REDUCE_H
