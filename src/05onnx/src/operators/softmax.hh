#ifndef ONNX_SOFTMAX_HH
#define ONNX_SOFTMAX_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Softmax final : public Operator {
        Int axis;

        explicit Softmax(Int);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SOFTMAX_HH
