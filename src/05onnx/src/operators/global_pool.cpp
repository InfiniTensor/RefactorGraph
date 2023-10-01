
#include "computation/operators/global_pool.h"
#include "common.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferGlobalPool(Operator const &, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (!input.dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto rank = input.rank();
        if (rank < 2) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        Shape output(rank, DimExpr(1));
        output[0] = input.shape[0];
        output[1] = input.shape[1];
        return Ok(Tensors{Tensor::share(input.dataType, std::move(output), extractDependency(inputs))});
    }

    LowerOperator lowerGlobalPool(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::GlobalAveragePool") ? PoolType::Average
                    : op.opType.is("onnx::GlobalLpPool")    ? PoolType::Lp
                    : op.opType.is("onnx::GlobalMaxPool")   ? PoolType::Max
                                                            : UNREACHABLEX(PoolType,
                                                                           "{} not support",
                                                                           op.opType.name());
        return {std::make_shared<GlobalPool>(type), {0}};
    }

}// namespace refactor::onnx
