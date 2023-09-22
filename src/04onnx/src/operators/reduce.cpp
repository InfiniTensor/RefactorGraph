#include "common/range.h"
#include "common.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferReduce(Operator const &op, Tensors inputs) {
        if (inputs.empty() || 2 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &data = inputs[0];
            if (!data->dataType.isNumberic()) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto keepdims = op.attribute("keepdims", {1}).int_();
            if (inputs.size() == 1) {
                if (op.attribute("noop_with_empty_axes", {0}).int_() != 0) {
                    return Ok(std::move(inputs));
                } else if (keepdims) {
                    return Ok(Tensors{Tensor::share(data->dataType,
                                                    Shape(data->shape.size(), DimExpr(1)),
                                                    extractDependency(inputs))});
                } else {
                    return Ok(Tensors{Tensor::share(data->dataType,
                                                    Shape{},
                                                    extractDependency(inputs))});
                }
            }
            auto const &axes = inputs[1];
            if (axes->dataType != DataType::I64 ||
                axes->shape.size() != 1 ||
                !axes->hasData()) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            auto axes_ = reinterpret_cast<int64_t *>(axes->data->ptr);
            auto const &shape = data->shape;
            EXPECT_VAL(axes->shape[0], axesSize)
            std::unordered_set<int64_t> axes__;
            for (auto i : range0_(axesSize)) {
                auto axis = axes_[i];
                axes__.insert(axis < 0 ? axis + shape.size() : axis);
            }
            Shape output;
            for (auto i : range0_(shape.size())) {
                if (axes__.find(i) == axes__.end()) {
                    output.emplace_back(shape[i]);
                } else if (keepdims) {
                    output.emplace_back(1);
                }
            }
            return Ok(Tensors{Tensor::share(data->dataType,
                                            std::move(output),
                                            extractDependency(inputs))});
        }
    }

    computation::SharedOp lowerReduce(Operator const &) {
        return nullptr;
    }
}// namespace refactor::onnx
