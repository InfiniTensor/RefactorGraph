#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    template<class TS, class TD>
    void castData(void *src, void *dst, size_t size) {
        auto src_ = reinterpret_cast<TS *>(src);
        auto dst_ = reinterpret_cast<TD *>(dst);
        std::transform(src_, src_ + size, dst_, [](auto x) { return static_cast<TD>(x); });
    }

    InferResult inferCast(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &input = inputs[0];
            auto output = input->shape;
            auto to = static_cast<DataType>(op.attribute("to").int_());
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output))});
            }
            auto from = input->dataType;
            if (from == to) {
                return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), input->data)});
            }
            fmt::print("({} -> {})", dataTypeName(from), dataTypeName(to));
            auto size = sizeOf(output);
            auto eleSize = dataTypeSize(to);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            switch (from) {
                case DataType::F32:
                    switch (to) {
                        case DataType::I64:
                            castData<float, int64_t>(input->data->ptr, blob->ptr, size);
                            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), std::move(blob))});

                        default:
                            break;
                    }
                    break;

                case DataType::I64: {
                    switch (to) {
                        case DataType::F32:
                            castData<int64_t, float>(input->data->ptr, blob->ptr, size);
                            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), std::move(blob))});

                        case DataType::Bool:
                            castData<int64_t, bool>(input->data->ptr, blob->ptr, size);
                            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output), std::move(blob))});

                        default:
                            break;
                    }
                } break;

                default:
                    break;
            }
            return Ok(Tensors{std::make_shared<Tensor>(to, std::move(output))});
        }
    }
}// namespace refactor::onnx
