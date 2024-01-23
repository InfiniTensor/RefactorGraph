#ifndef LLM_RMS_NORMALIZATION_HH
#define LLM_RMS_NORMALIZATION_HH

#include "frontend/operator.h"

namespace refactor::llm {
    using namespace frontend;

    struct RmsNormalization final : public Operator {

        constexpr RmsNormalization() noexcept = default;

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::llm

#endif// LLM_RMS_NORMALIZATION_HH
