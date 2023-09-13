#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferWhere(Operator const &op, Edges inputs) {
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
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(output))});
            }

            auto [shape, size] = shape_size(output);
            auto eleSize = dataTypeSize(dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto dst = reinterpret_cast<uint8_t *>(blob->ptr);
            auto srcC = reinterpret_cast<bool *>(condition->data->ptr);
            auto srcX = reinterpret_cast<uint8_t *>(x->data->ptr);
            auto srcY = reinterpret_cast<uint8_t *>(y->data->ptr);
            fmt::print("( {} dst<{}> = ", op.opType.name(), size);
            for (size_t i = 0; i < size; ++i) {
                auto indices = buildIndices(shape, i);
                auto ptr = [&indices](Edge const &input) -> uint8_t * {
                    auto it0 = indices.rbegin(),
                         end0 = indices.rend();
                    auto it1 = input->shape.rbegin(),
                         end1 = input->shape.rend();
                    size_t ii = 0, mul = 1;
                    while (it0 != end0 && it1 != end1) {
                        ii += *it0++ * mul;
                        mul *= it1++->value();
                    }
                    auto src = reinterpret_cast<uint8_t *>(input->data->ptr);
                    return reinterpret_cast<uint8_t *>(src + ii * dataTypeSize(input->dataType));
                };

                if (*reinterpret_cast<bool *>(ptr(condition))) {
                    std::copy_n(ptr(x), eleSize, dst);
                    fmt::print("x ");
                } else {
                    std::copy_n(ptr(y), eleSize, dst);
                    fmt::print("y ");
                }
                dst += eleSize;
            }
            fmt::print(")");
            return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(output), std::move(blob))});
        }
    }
}// namespace refactor::onnx
