#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferArithmetic(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto dataType = inputs[0]->dataType;
            if (!isNumbericDataType(dataType) || inputs[1]->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            } else {
                auto res = multidirBroadcast({inputs[0]->shape, inputs[1]->shape});
                if (res.isErr()) {
                    return Err(InferError(ERROR_MSG(res.unwrapErr())));
                }
                auto output = std::move(res.unwrap());
                if (!shouldCalculate(inputs, output) || dataType != DataType::I64) {
                    return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(output))});
                }

                auto ss = output.size();
                auto [shape, size] = shape_size(output);
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                auto dst = reinterpret_cast<uint64_t *>(blob->ptr);
                auto opType = op.opType.name();
                fmt::print("{} dst[{}] = ( ", opType, size);
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
                    if (opType == "onnx::Add") {
                        dst[i] = a + b;
                    } else if (opType == "onnx::Sub") {
                        dst[i] = a - b;
                    } else if (opType == "onnx::Mul") {
                        dst[i] = a * b;
                    } else if (opType == "onnx::Div") {
                        dst[i] = a / b;
                    } else {
                        return Err(InferError(ERROR_MSG("OpType not support")));
                    }
                    fmt::print("{} ", dst[i]);
                }
                fmt::println(")");
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(output), std::move(blob))});
            }
        }
    }
}// namespace refactor::onnx
