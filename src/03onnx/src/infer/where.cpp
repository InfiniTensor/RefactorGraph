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
            auto output = std::move(res.unwrap());
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{std::make_shared<Tensor>(dataType, std::move(output))});
            }

            auto [shape, size] = shape_size(output);
            auto eleSize = dataTypeSize(dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto dst = reinterpret_cast<uint8_t *>(blob->ptr);
            fmt::print("( {} dst<{}> = ", op.opType.name(), size);
            for (size_t i = 0; i < size; ++i) {
                auto indices = buildIndices(shape, i);
                if (*reinterpret_cast<bool *>(locate(*condition, indices))) {
                    std::memcpy(dst + i * eleSize, locate(*x, indices), eleSize);
                    fmt::print("x ");
                } else {
                    std::memcpy(dst + i * eleSize, locate(*y, indices), eleSize);
                    fmt::print("y ");
                }
            }
            fmt::print(")");
            return Ok(Tensors{std::make_shared<Tensor>(dataType, std::move(output), std::move(blob))});
        }
    }
}// namespace refactor::onnx
