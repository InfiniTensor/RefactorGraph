#include "common/range.h"
#include "infer.h"
#include <execution>

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
        for (auto i : range(1ul, inputs.size())) {
            auto const &input = inputs[i];
            if (input->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (input->shape.size() != output.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            for (auto i : range0_(output.size())) {
                if (i == axis) {
                    EXPECT_VAL(output[i], a)
                    EXPECT_VAL(input->shape[i], b)
                    output[i] = DimExpr(a + b);
                } else if (output[i] != input->shape[i]) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
        }
        auto ans = Tensor::share(dataType, std::move(output));
        if (!shouldCalculate(inputs, output)) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::par_unseq, natural_t(0), ans->elementsSize(),
                        [&,
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = dataTypeSize(dataType)](auto const i) {
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
                            std::memcpy(dst + i * eleSize, locate1(*inputs[k], indices), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }
}// namespace refactor::onnx
