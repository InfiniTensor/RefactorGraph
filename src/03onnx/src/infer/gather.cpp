#include "common/range.h"
#include "infer.h"
#include <execution>

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
            auto axis = op.attribute("axis", {0}).int_();
            if (axis < 0) {
                axis += r;
            }
            if (axis < 0 || r <= axis) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto output = data->shape;
            output.erase(output.begin() + axis);
            output.insert(output.begin() + axis, indices->shape.begin(), indices->shape.end());
            auto ans = Tensor::share(data->dataType, std::move(output), extractDependency(inputs));
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{std::move(ans)});
            }

            std::for_each_n(std::execution::par_unseq, natural_t(0), ans->elementsSize(),
                            [&data, &indices, &output,
                             axis,
                             q = indices->shape.size(),
                             ssz = output.size(),
                             src = reinterpret_cast<uint8_t *>(data->data->ptr),
                             dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                             eleSize = dataTypeSize(data->dataType)](auto const i) {
                                auto indices_ = locateN(output, i);
                                int64_t k;
                                {
                                    size_t ii = 0, mul = 1;
                                    for (auto j : range0_(q).rev()) {
                                        ii += indices_[j] * mul;
                                        mul *= indices->shape[j].value();
                                    }
                                    k = indices->dataType == DataType::I64
                                            ? reinterpret_cast<int64_t *>(indices->data->ptr)[ii]
                                            : reinterpret_cast<int32_t *>(indices->data->ptr)[ii];
                                }
                                {
                                    size_t ii = 0, mul = 1;
                                    for (auto j : range(static_cast<decltype(q)>(axis) + q, ssz).rev()) {
                                        ii += indices_[j] * mul;
                                        mul *= data->shape[j - q + 1].value();
                                    }
                                    ii += k * mul;
                                    for (auto j : range0_(axis).rev()) {
                                        ii += indices_[j] * mul;
                                        mul *= data->shape[j].value();
                                    }
                                    std::memcpy(dst + i * eleSize, src + ii * eleSize, eleSize);
                                }
                            });

            return Ok(Tensors{std::move(ans)});
        }
    }

}// namespace refactor::onnx
