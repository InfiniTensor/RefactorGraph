#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferCast(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto to = static_cast<DataType>(op.attribute("to").int_());
            auto const &input = inputs[0];
            auto output = input->shape;
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output))});
            }
            auto from = input->dataType;
            if (from == to) {
                return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), input->data)});
            }
            fmt::print("({} -> {})", dataTypeName(from), dataTypeName(to));
            auto [shape, size] = shape_size(output);
            auto eleSize = dataTypeSize(to);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            switch (from) {
                case DataType::F32:
                    switch (to) {
                        case DataType::I64: {
                            auto dst = reinterpret_cast<int64_t *>(blob->ptr);
                            auto src = reinterpret_cast<float *>(input->data->ptr);
                            std::transform(src, src + size, dst, [](auto x) { return static_cast<int64_t>(x); });
                            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), std::move(blob))});
                        }
                        default:
                            break;
                    }
                    break;

                case DataType::I64:
                    switch (to) {
                        case DataType::F32: {
                            auto dst = reinterpret_cast<float *>(blob->ptr);
                            auto src = reinterpret_cast<int64_t *>(input->data->ptr);
                            std::transform(src, src + size, dst, [](auto x) { return static_cast<float>(x); });
                            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), std::move(blob))});
                        }
                        default:
                            break;
                    }
                    break;

                default:
                    break;
            }
            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output))});
        }
    }
}// namespace refactor::onnx
