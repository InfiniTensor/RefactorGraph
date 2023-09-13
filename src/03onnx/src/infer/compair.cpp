#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferEqual(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (a->dataType != b->dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }

            auto res = multidirBroadcast({a->shape, b->shape});
            if (res.isErr()) {
                return Err(InferError(ERROR_MSG(res.unwrapErr())));
            }

            auto dataType = a->dataType;
            auto output = std::move(res.unwrap());
            if (!shouldCalculate(inputs, output) || dataType != DataType::I64) {
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(output))});
            }

            auto [shape, size] = shape_size(output);
            auto eleSize = dataTypeSize(dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto dst = reinterpret_cast<bool *>(blob->ptr);
            fmt::print("( {} dst<{}> = ", op.opType.name(), size);
            for (size_t i = 0; i < size; ++i) {
                auto indices = buildIndices(shape, i);
                auto getter = [&indices](Edge const &input) -> int64_t {
                    auto it0 = indices.rbegin(),
                         end0 = indices.rend();
                    auto it1 = input->shape.rbegin(),
                         end1 = input->shape.rend();
                    size_t ii = 0, mul = 1;
                    while (it0 != end0 && it1 != end1) {
                        ii += *it0++ * mul;
                        mul *= it1++->value();
                    }
                    return reinterpret_cast<int64_t *>(input->data->ptr)[ii];
                };

                auto a = getter(inputs[0]), b = getter(inputs[1]);
                dst[i] = a == b;
                fmt::print("{} ", dst[i]);
            }
            fmt::print(")");
            return Ok(Edges{std::make_shared<Tensor>(DataType::Bool, std::move(output), std::move(blob))});
        }
    }
}// namespace refactor::onnx
