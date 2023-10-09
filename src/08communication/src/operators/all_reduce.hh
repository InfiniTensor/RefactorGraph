#ifndef COMMUNICATION_ALL_REDUCE_HH
#define COMMUNICATION_ALL_REDUCE_HH

#include "frontend/operator.h"

namespace refactor::communication {
    using namespace frontend;

    struct AllReduce final : public Operator {

        AllReduce();

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::communication

#endif// COMMUNICATION_ALL_REDUCE_HH
