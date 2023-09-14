#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferCumSum(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2) {
            auto const &x = inputs[0];
            auto const &axis = inputs[1];
            if (!axis->shape.empty()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (!isNumbericDataType(x->dataType) ||
                (axis->dataType != DataType::I64 &&
                 axis->dataType != DataType::I32)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            return Ok(Tensors{Tensor::share(x->dataType, x->shape)});
        }
    }
}// namespace refactor::onnx
