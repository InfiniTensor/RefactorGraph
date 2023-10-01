#include "common.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferBatchNormalization(Operator const &op, TensorRefs inputs, InferOptions const &) {
        if (op.attribute("training_mode", {0}).int_() != 0) {
            RUNTIME_ERROR("training_mode is not supported");
        }

        EXPECT_SIZE(5)

        auto const &x = inputs[0];
        auto const &scale = inputs[1];
        auto const &bias = inputs[2];
        auto const &mean = inputs[3];
        auto const &var = inputs[4];

        if (!x.dataType.isFloat() ||
            !scale.dataType.isFloat() || bias.dataType != scale.dataType ||
            !mean.dataType.isFloat() || var.dataType != mean.dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        if (x.rank() <= 2 ||
            bias.shape != scale.shape ||
            mean.shape != scale.shape ||
            var.shape != scale.shape) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }

        return Ok(Tensors{Tensor::share(x.dataType, x.shape, extractDependency(inputs))});
    }

    LowerOperator lowerBatchNormalization(Operator const &, TensorRefs) {
        UNREACHABLE();
    }

}// namespace refactor::onnx
