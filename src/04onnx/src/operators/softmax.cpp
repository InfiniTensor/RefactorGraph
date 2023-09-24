#include "computation/operators/softmax.h"
#include "common.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferSoftmax(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1)
        if (!inputs[0]->dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        return Ok(std::move(inputs));
    }

    computation::SharedOp lowerSoftmax(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto axis = op.attribute("axis", {-1}).int_();
        if (axis < 0) {
            axis += inputs[0].rank();
        }
        return std::make_shared<Softmax>(static_cast<size_t>(axis));
    }
}// namespace refactor::onnx
