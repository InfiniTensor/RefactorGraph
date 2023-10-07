#ifndef ONNX_GEMM_HH
#define ONNX_GEMM_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Gemm final : public Operator {
        Float alpha, beta;
        bool transA, transB;

        Gemm(Float, Float, bool, bool);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_GEMM_HH
