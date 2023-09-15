#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferWhere(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(3) {
            auto const &condition = inputs[0];
            auto const &x = inputs[1];
            auto const &y = inputs[2];
            if (condition->dataType != DataType::Bool) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (x->dataType != y->dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }

            auto res = multidirBroadcast({condition->shape, x->shape, y->shape});
            if (res.isErr()) {
                return Err(InferError(ERROR_MSG(res.unwrapErr())));
            }
            auto dataType = x->dataType;
            auto ans = Tensor::share(dataType, std::move(res.unwrap()));
            if (!shouldCalculate(inputs, ans->shape)) {
                return Ok(Tensors{std::move(ans)});
            }

            auto size = ans->elementsSize();
            auto eleSize = dataTypeSize(dataType);
            auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
            for (size_t i = 0; i < size; ++i) {
                auto indices = locateN(ans->shape, i);
                if (*reinterpret_cast<bool *>(locate1(*condition, indices))) {
                    std::memcpy(dst + i * eleSize, locate1(*x, indices), eleSize);
                } else {
                    std::memcpy(dst + i * eleSize, locate1(*y, indices), eleSize);
                }
            }
            return Ok(Tensors{std::move(ans)});
        }
    }
}// namespace refactor::onnx
