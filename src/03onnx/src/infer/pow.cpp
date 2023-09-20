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
            auto res = multidirBroadcast({a->shape, b->shape});
            if (res.isErr()) {
                return Err(InferError(ERROR_MSG(res.unwrapErr())));
            }
            auto ans = Tensor::share(a->dataType, std::move(res.unwrap()));
            return Ok(Tensors{std::move(ans)});
        }
    }
}// namespace refactor::onnx
