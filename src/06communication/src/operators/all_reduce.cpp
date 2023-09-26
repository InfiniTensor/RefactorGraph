#include "common.h"

namespace refactor::communication {
    using namespace frontend;

    InferResult inferAllReduce(Operator const &op, TensorRefs inputs) {
        EXPECT_SIZE(1) {
            return Ok(Tensors{
                Tensor::share(inputs[0].dataType, inputs[0].shape, extractDependency(inputs))});
        }
    }

    computation::SharedOp lowerAllReduce(Operator const &op, TensorRefs) {
        return nullptr;
    }
}// namespace refactor::communication
