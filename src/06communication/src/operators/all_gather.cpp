#include "common.h"

namespace refactor::communication {
    using namespace frontend;

    InferResult inferAllGather(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1) {
            return Ok(Tensors(
                op.attribute("nranks").int_(),
                Tensor::share(inputs[0].dataType, inputs[0].shape, extractDependency(inputs))));
        }
    }

}// namespace refactor::communication
