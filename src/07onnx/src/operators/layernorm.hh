#ifndef ONNX_LAYERNORM_HH
#define ONNX_LAYERNORM_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Layernorm final : public Operator {
        Int axis;
        Float epsilon;

        explicit Layernorm(Int, Float);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_LAYERNORM_HH