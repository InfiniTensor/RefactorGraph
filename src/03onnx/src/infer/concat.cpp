#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferConcat(Operator const &op, Tensors inputs) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto dataType = inputs[0]->dataType;
        auto output = inputs[0]->shape;
        auto rank = output.size();
        auto axis = op.attribute("axis").int_();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            auto const &input = *it;
            if (input->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (input->shape.size() != output.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            for (size_t i = 0; i < output.size(); ++i) {
                if (i == axis) {
                    EXPECT_VAL(output[i], a)
                    EXPECT_VAL(input->shape[i], b)
                    output[i] = DimExpr(a + b);
                } else if (output[i] != input->shape[i]) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
        }
        if (!shouldCalculate(inputs, output)) {
            return Ok(Tensors{std::make_shared<Tensor>(dataType, std::move(output))});
        }

        auto size = sizeOf(output);
        auto eleSize = dataTypeSize(dataType);
        auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
        auto dst = reinterpret_cast<uint8_t *>(blob->ptr);
        for (size_t i = 0; i < size; ++i) {
            auto indices = locateN(output, i);

            size_t k = 0;
            for (auto axis_ = indices[axis]; k < inputs.size(); ++k) {
                auto axis__ = inputs[k]->shape[axis].value();
                if (axis_ >= axis__) {
                    axis_ -= axis__;
                } else {
                    indices[axis] = axis_;
                    break;
                }
            }
            size_t ii = 0, mul = 1;
            for (size_t j = 0; j < rank; ++j) {
                auto j_ = rank - 1 - j;// reverse
                ii += indices[j_] * mul;
                mul *= inputs[k]->shape[j_].value();
            }
            auto input = reinterpret_cast<uint8_t *>(inputs[k]->data->ptr);
            std::memcpy(dst + i * eleSize, input + ii * eleSize, eleSize);
        }
        return Ok(Tensors{std::make_shared<Tensor>(dataType, std::move(output), std::move(blob))});
    }
}// namespace refactor::onnx
