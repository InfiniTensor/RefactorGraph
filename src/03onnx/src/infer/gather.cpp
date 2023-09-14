#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferGather(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2)
        if (inputs[1]->dataType != DataType::I32 && inputs[1]->dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto const &data = inputs[0];
            auto const &indices = inputs[1];
            auto const r = data->shape.size();
            auto const q = indices->shape.size();
            auto axis = op.attribute("axis", {0}).int_();
            if (axis < 0) {
                axis += r;
            }
            if (axis < 0 || r <= axis) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto dataType = data->dataType;
            auto output = data->shape;
            output.erase(output.begin() + axis);
            output.insert(output.begin() + axis, indices->shape.begin(), indices->shape.end());
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{Tensor::share(dataType, std::move(output))});
            }

            auto const ssz = output.size();
            auto size = sizeOf(output);
            auto eleSize = dataTypeSize(dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto src = reinterpret_cast<uint8_t *>(data->data->ptr);
            auto dst = reinterpret_cast<uint8_t *>(blob->ptr);

            for (size_t i = 0; i < size; ++i) {
                auto indices_ = locateN(output, i);
                int64_t k;
                {
                    size_t ii = 0, mul = 1;
                    for (size_t j = 0; j < q; ++j) {
                        auto j_ = q - 1 - j;// reverse
                        ii += indices_[j_] * mul;
                        mul *= indices->shape[j_].value();
                    }
                    k = indices->dataType == DataType::I64
                            ? reinterpret_cast<int64_t *>(indices->data->ptr)[ii]
                            : reinterpret_cast<int32_t *>(indices->data->ptr)[ii];
                }
                {
                    size_t ii = 0, mul = 1;
                    for (size_t j = axis + q; j < ssz; ++j) {
                        auto j_ = ssz - 1 - j;// reverse
                        ii += indices_[j_] * mul;
                        mul *= data->shape[j_ - q + 1].value();
                    }
                    ii += k * mul;
                    for (size_t j = 0; j < axis; ++j) {
                        auto j_ = axis - 1 - j;// reverse
                        ii += indices_[j_] * mul;
                        mul *= data->shape[j_].value();
                    }
                    std::memcpy(dst + i * eleSize, src + ii * eleSize, eleSize);
                }
                // fmt::println("gather copies {} bytes", eleSize);
            }

            return Ok(Tensors{Tensor::share(dataType, std::move(output), std::move(blob))});
        }
    }

}// namespace refactor::onnx
