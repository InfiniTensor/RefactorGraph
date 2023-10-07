#ifndef ONNX_RANGE_HH
#define ONNX_RANGE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Range final : public Operator {
        Range();

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_RANGE_HH
