#ifndef ONNX_CLIP_HH
#define ONNX_CLIP_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Clip final : public Operator {

        Clip() = default;

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CLIP_HH
