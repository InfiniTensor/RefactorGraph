#ifndef ONNX_CONV_HH
#define ONNX_CONV_HH

#include "common.h"
#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Conv final : public Operator {
        OptionalInts dilations, pads, strides;

        Conv(OptionalInts dilations,
             OptionalInts pads,
             OptionalInts strides);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CONV_HH
