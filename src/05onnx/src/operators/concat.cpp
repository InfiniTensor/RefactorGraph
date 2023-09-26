#include "computation/operators/concat.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferConcat(Operator const &op, TensorRefs inputs, InferOptions const& options) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto dataType = inputs[0].dataType;
        auto output = inputs[0].shape;
        auto rank = inputs[0].rank();
        auto axis = op.attribute("axis").int_();
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || rank <= axis) {
            return Err(InferError(ERROR_MSG("Axis out of range")));
        }
        for (auto i : range(1ul, inputs.size())) {
            auto const &input = inputs[i];
            if (input.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (input.shape.size() != output.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            for (auto i : range0_(static_cast<int64_t>(output.size()))) {
                if (i == axis) {
                    EXPECT_VAL(output[i], a)
                    EXPECT_VAL(input.shape[i], b)
                    output[i] = DimExpr(a + b);
                } else if (output[i] != input.shape[i]) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
        }
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&,
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = dataType.size()](auto const i) {
                            auto indices = locateN(output, i);

                            size_t k = 0;
                            for (auto axis_ = indices[axis]; k < inputs.size(); ++k) {
                                auto axis__ = inputs[k].shape[axis].value();
                                if (axis_ >= axis__) {
                                    axis_ -= axis__;
                                } else {
                                    indices[axis] = axis_;
                                    break;
                                }
                            }
                            std::memcpy(dst + i * eleSize, locate1(inputs[k], indices), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    computation::SharedOp lowerConcat(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto axis = op.attribute("axis").int_();
        if (axis < 0) {
            axis += inputs[0].rank();
        }
        return std::make_shared<Concat>(static_cast<size_t>(axis));
    }
}// namespace refactor::onnx
