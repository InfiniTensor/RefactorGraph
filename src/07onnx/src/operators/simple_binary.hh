#ifndef ONNX_SIMPLE_BINARY_HH
#define ONNX_SIMPLE_BINARY_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class SimpleBinaryType {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        And,
        Or,
        Xor,
    };

    struct SimpleBinary final : public Operator {
        SimpleBinaryType type;

        explicit SimpleBinary(SimpleBinaryType);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(SimpleBinaryType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SIMPLE_BINARY_HH
