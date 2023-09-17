#include "common/range.h"
#include "infer.h"
#include <execution>

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

            auto eleSize = dataTypeSize(dataType);
            auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), ans->elementsSize(), [&, dst, eleSize](auto i) {
                                auto indices = locateN(ans->shape, i);
                                auto const &tensor = *reinterpret_cast<bool *>(locate1(*condition, indices))
                                                         ? x
                                                         : y;
                                std::memcpy(dst + i * eleSize, locate1(*tensor, indices), eleSize);
                            });
            return Ok(Tensors{std::move(ans)});
        }
    }
}// namespace refactor::onnx
