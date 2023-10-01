#include "computation/operators/softmax.h"
#include "common.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferSoftmax(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1)
        if (!inputs[0].dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        return Ok(Tensors{Tensor::share(inputs[0])});
    }

    LowerOperator lowerSoftmax(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto rank = inputs[0].rank();
        auto axis = op.attribute("axis", {-1}).int_();
        return {std::make_shared<Softmax>(axis < 0 ? axis + rank : axis, rank), {0}};
    }
}// namespace refactor::onnx
