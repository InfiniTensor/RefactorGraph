#ifndef ONNX_TRANSPOSE_HH
#define ONNX_TRANSPOSE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Transpose final : public Operator {
        Ints perm;

        explicit Transpose(Ints);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_TRANSPOSE_HH
