#ifndef ONNX_SPLIT_HH
#define ONNX_SPLIT_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Split final : public Operator {
        Int axis, numOutputs;

        Split(Int, Int);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SPLIT_HH
