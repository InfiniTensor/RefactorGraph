#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferSoftmax(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1)
        if (!isIeee754DataType(inputs[0]->dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            return Ok(std::move(inputs));
        }
    }
}// namespace refactor::onnx
