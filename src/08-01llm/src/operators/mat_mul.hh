#ifndef LLM_MAT_MUL_HH
#define LLM_MAT_MUL_HH

#include "frontend/operator.h"

namespace refactor::llm {
    using namespace frontend;

    struct MatMul final : public Operator {
        bool transA, transB;

        MatMul(decltype(transA), decltype(transB));

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::llm

#endif// LLM_MAT_MUL_HH
