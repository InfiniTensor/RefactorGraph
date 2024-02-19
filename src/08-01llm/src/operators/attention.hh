#ifndef LLM_RMS_ATTENTION_HH
#define LLM_RMS_ATTENTION_HH

#include "frontend/operator.h"

namespace refactor::llm {
    using namespace frontend;

    struct Attention final : public Operator {
        dim_t maxSeqLen;

        explicit Attention(decltype(maxSeqLen));

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::llm

#endif// LLM_RMS_ATTENTION_HH
