#ifndef ONNX_SIMPLE_UNARY_HH
#define ONNX_SIMPLE_UNARY_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class SimpleUnaryType {
        Abs,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Cos,
        Cosh,
        Erf,
        HardSwish,
        Identity,
        Log,
        Not,
        Neg,
        Relu,
        Sin,
        Sinh,
        Sqrt,
        Sigmoid,
        Tan,
        Tanh,
    };

    struct SimpleUnary final : public Operator {
        SimpleUnaryType type;

        explicit SimpleUnary(SimpleUnaryType);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(SimpleUnaryType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SIMPLE_UNARY_HH
