#ifndef ONNX_CONSTANT_HH
#define ONNX_CONSTANT_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Constant final : public Operator {
        Attribute value;

        explicit Constant(Attribute);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CONSTANT_HH
