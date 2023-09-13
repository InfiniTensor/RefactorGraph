#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferCompair(Operator const &op, Tensors inputs) {
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

            auto output = std::move(res.unwrap());
            if (!shouldCalculate(inputs, output) || a->dataType != DataType::I64) {
                return Ok(Tensors{std::make_shared<Tensor>(a->dataType, std::move(output))});
            }

            auto [shape, size] = shape_size(output);
            auto eleSize = dataTypeSize(DataType::Bool);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto dst = reinterpret_cast<bool *>(blob->ptr);
            auto opType = op.opType.name();
            fmt::print("( {} dst<{}> = ", opType, size);
            for (size_t i = 0; i < size; ++i) {
                auto indices = buildIndices(shape, i);
                auto a_ = *reinterpret_cast<int64_t *>(locate(*a, indices)),
                     b_ = *reinterpret_cast<int64_t *>(locate(*b, indices));
                if (opType == "onnx::Equal") {
                    dst[i] = a_ == b_;
                } else if (opType == "onnx::Greater") {
                    dst[i] = a_ > b_;
                } else if (opType == "onnx::GreaterOrEqual") {
                    dst[i] = a_ >= b_;
                } else if (opType == "onnx::Less") {
                    dst[i] = a_ < b_;
                } else if (opType == "onnx::LessOrEqual") {
                    dst[i] = a_ <= b_;
                } else {
                    return Err(InferError(ERROR_MSG("OpType not support")));
                }
                fmt::print("{} ", dst[i]);
            }
            fmt::print(")");
            return Ok(Tensors{std::make_shared<Tensor>(DataType::Bool, std::move(output), std::move(blob))});
        }
    }
}// namespace refactor::onnx
