#ifndef ONNX_SLICE_HH
#define ONNX_SLICE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Slice final : public Operator {
        Slice();

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SLICE_HH
