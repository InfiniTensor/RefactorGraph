#ifndef COMMUNICATION_ALL_REDUCE_HH
#define COMMUNICATION_ALL_REDUCE_HH

#include "frontend/operator.h"
#include "kernel/attributes/communication.h"

namespace refactor::communication {
    using namespace frontend;

    struct AllReduce final : public Operator {
        kernel::AllReduceType type;

        AllReduce(kernel::AllReduceType);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(kernel::AllReduceType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::communication

#endif// COMMUNICATION_ALL_REDUCE_HH
