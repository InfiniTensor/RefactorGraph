#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferCumSum(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (!inputs[1]->shape.empty()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        } else if (!isNumbericDataType(inputs[0]->dataType) ||
                   (inputs[1]->dataType != DataType::I64 &&
                    inputs[1]->dataType != DataType::I32)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            return Ok(Edges{std::move(inputs[0])});
        }
    }
}// namespace refactor::onnx
