#ifndef ONNX_CONSTANT_OF_SHAPE_HH
#define ONNX_CONSTANT_OF_SHAPE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct ConstantOfShape final : public Operator {
        std::shared_ptr<Tensor> value;

        explicit ConstantOfShape(std::shared_ptr<Tensor>);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CONSTANT_OF_SHAPE_HH
