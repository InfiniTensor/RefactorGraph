#ifndef COMMUNICATION_ALL_GATHER_HH
#define COMMUNICATION_ALL_GATHER_HH

#include "frontend/operator.h"

namespace refactor::communication {
    using namespace frontend;

    struct AllGather final : public Operator {
        Int nranks;

        explicit AllGather(Int);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::communication

#endif// COMMUNICATION_ALL_GATHER_HH
