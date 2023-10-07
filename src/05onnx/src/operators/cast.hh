#ifndef ONNX_CAST_HH
#define ONNX_CAST_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Cast final : public Operator {
        common::DataType to;

        explicit Cast(common::DataType);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CAST_HH
