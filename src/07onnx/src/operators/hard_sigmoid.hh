#ifndef ONNX_HARD_SIGMOID_HH
#define ONNX_HARD_SIGMOID_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct HardSigmoid final : public Operator {
        Float alpha, beta;

        explicit HardSigmoid(Float, Float);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_HARD_SIGMOID_HH
