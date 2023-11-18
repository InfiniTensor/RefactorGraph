#ifndef ONNX_SHAPE_HH
#define ONNX_SHAPE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Shape final : public Operator {
        Int start;
        std::optional<Int> end;

        Shape(Int, std::optional<Int>);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SHAPE_HH
