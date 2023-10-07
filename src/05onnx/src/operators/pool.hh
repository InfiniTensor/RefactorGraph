#ifndef ONNX_POOL_HH
#define ONNX_POOL_HH

#include "frontend/operator.h"
#include "pool_type.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Pool final : public Operator {
        PoolType type;
        Ints kernelShape;
        OptionalInts dilations, pads, strides;

        explicit Pool(PoolType,
                      Ints kernelShape,
                      OptionalInts dilations,
                      OptionalInts pads,
                      OptionalInts strides);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId(PoolType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_POOL_HH
