#ifndef ONNX_TOPK_HH
#define ONNX_TOPK_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct TopK final : public Operator {
        Int topk, axis;
        TopK(Int topk, Int axis);

        static size_t typeId();
        static OpBox build(ModelContext const &, std::string_view, Attributes);
        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_WHERE_HH
