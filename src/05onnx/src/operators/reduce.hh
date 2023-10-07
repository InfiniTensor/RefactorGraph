#ifndef ONNX_REDUCE_HH
#define ONNX_REDUCE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class ReduceType {
        Mean,
        L1,
        L2,
        LogSum,
        LogSumExp,
        Max,
        Min,
        Prod,
        Sum,
        SumSquare,
    };

    struct Reduce final : public Operator {
        ReduceType type;
        bool keepdims, noopWithEmptyAxes;

        Reduce(ReduceType, bool, bool);

        static OpBox build(std::string_view, Attributes);
        static size_t typeId(ReduceType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        LowerOperator lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_REDUCE_HH
