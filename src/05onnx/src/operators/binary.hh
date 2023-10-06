#ifndef ONNX_BINARY_HH
#define ONNX_BINARY_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class BinaryType {
        Add,
        Sub,
        Mul,
        Div,
    };

    struct Binary final : public Operator {
        BinaryType type;

        explicit Binary(BinaryType);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId(BinaryType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_BINARY_HH
