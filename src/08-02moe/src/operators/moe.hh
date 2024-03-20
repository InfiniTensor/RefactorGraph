#ifndef MOE_HH
#define MOE_HH

#include "frontend/operator.h"

namespace refactor::moe {
    using namespace frontend;

    struct AssignPos final : public Operator {
        uint32_t topk, numExperts;
        explicit AssignPos(uint32_t topk, uint32_t numExperts);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

    struct Reorder final : public Operator {
        bool scatter;  
        uint32_t top, dim;
        explicit Reorder(bool scatter, uint32_t topk, uint32_t dim);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::llm

#endif// LLM_RMS_ATTENTION_HH
