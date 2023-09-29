#include "computation/operators/gather.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferGather(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &indices = inputs[1];
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        auto const r = data.rank();
        auto axis = op.attribute("axis", {0}).int_();
        if (axis < 0) {
            axis += r;
        }
        if (axis < 0 || r <= axis) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto output = data.shape;
        output.erase(output.begin() + axis);
        output.insert(output.begin() + axis, indices.shape.begin(), indices.shape.end());
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&data, &indices, &output,
                         axis,
                         q = indices.shape.size(),
                         ssz = output.size(),
                         src = data.data->get<uint8_t>(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = data.dataType.size()](auto const i) {
                            auto indices_ = locateN(output, i);
                            int64_t k;
                            {
                                size_t ii = 0, mul = 1;
                                for (auto j : range0_(q).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= indices.shape[j].value();
                                }
                                k = indices.dataType == DataType::I64
                                        ? indices.data->get<int64_t>()[ii]
                                        : indices.data->get<int32_t>()[ii];
                            }
                            {
                                size_t ii = 0, mul = 1;
                                for (auto j : range(static_cast<decltype(q)>(axis) + q, ssz).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= data.shape[j - q + 1].value();
                                }
                                ii += k * mul;
                                for (auto j : range0_(axis).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= data.shape[j].value();
                                }
                                std::memcpy(dst + i * eleSize, src + ii * eleSize, eleSize);
                            }
                        });

        return Ok(Tensors{std::move(ans)});
    }

    LowerOperator lowerGather(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto axis = op.attribute("axis", {0}).int_();
        if (axis < 0) {
            axis += inputs[0].rank();
        }
        return {std::make_shared<Gather>(static_cast<size_t>(axis)), {0, 1}};
    }
}// namespace refactor::onnx
