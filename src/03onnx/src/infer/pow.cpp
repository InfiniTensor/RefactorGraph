#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferPow(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2) {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (!isSignedDataType(a->dataType) || !isNumbericDataType(b->dataType)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto ans = multidirBroadcast({a->shape, b->shape});
            if (ans.isErr()) {
                return Err(InferError(ERROR_MSG(ans.unwrapErr())));
            } else {
                return Ok(Tensors{std::make_shared<Tensor>(a->dataType, ans.unwrap())});
            }
        }
    }
}// namespace refactor::onnx
