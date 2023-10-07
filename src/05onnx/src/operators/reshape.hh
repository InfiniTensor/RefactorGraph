#ifndef ONNX_RESHAPE_HH
#define ONNX_RESHAPE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Reshape final : public Operator {
        bool allowzero;

        explicit Reshape(bool);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_RESHAPE_HH
