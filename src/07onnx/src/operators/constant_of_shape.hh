#ifndef ONNX_CONSTANT_OF_SHAPE_HH
#define ONNX_CONSTANT_OF_SHAPE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct ConstantOfShape final : public Operator {
        Tensor_ value;

        explicit ConstantOfShape(Tensor_);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CONSTANT_OF_SHAPE_HH
