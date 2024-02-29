#ifndef LLM_ROPE_HH
#define LLM_ROPE_HH

#include "frontend/operator.h"

namespace refactor::llm {
    using namespace frontend;

    struct RotaryPositionEmbedding final : public Operator {
        float theta;

        explicit RotaryPositionEmbedding(float _theta);

        static OpBox build(ModelContext const &, std::string_view, Attributes);

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view opTypeName() const final;

        InferResult infer(TensorRefs, InferOptions const &) const final;

        computation::OpBox lower(TensorRefs) const final;

    };
}


#endif
