#ifndef ONNX_REDUCE_HH
#define ONNX_REDUCE_HH

#include "computation/operators/reduce.h"
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
        Ints axes;
        bool keepdims, noopWithEmptyAxes;

        Reduce(ReduceType, Ints, bool, bool);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(ReduceType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_REDUCE_HH
