#include "infer.h"

namespace refactor::communication {
    using namespace computation;

    InferResult inferAllReduce(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &input = inputs[0];
            return Ok(Tensors{Tensor::share(input->dataType, input->shape)});
        }
    }
}// namespace refactor::communication
