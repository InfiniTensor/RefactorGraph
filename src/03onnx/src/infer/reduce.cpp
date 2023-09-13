#include "infer.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferReduce(Operator const &op, Tensors inputs) {
        if (inputs.empty() || 2 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!isNumbericDataType(inputs[0]->dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto keepdims = op.attribute("keepdims", {1}).int_();
            if (inputs.size() == 2) {
                auto const &shape = inputs[0]->shape;
                auto const &axes = inputs[1];
                if (axes->dataType != DataType::I64 ||
                    axes->shape.size() != 1 ||
                    !axes->hasData()) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                auto axes_ = reinterpret_cast<int64_t *>(axes->data->ptr);
                EXPECT_VAL(axes->shape[0], axesSize)
                std::unordered_set<int64_t> axes__;
                for (size_t i = 0; i < axesSize; ++i) {
                    auto axis = axes_[i];
                    axes__.insert(axis < 0 ? axis + shape.size() : axis);
                }
                Shape ans;
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (axes__.find(i) == axes__.end()) {
                        ans.emplace_back(shape[i]);
                    } else if (keepdims) {
                        ans.emplace_back(1);
                    }
                }
                return Ok(Tensors{std::make_shared<Tensor>(inputs[0]->dataType, std::move(ans))});
            } else if (op.attribute("noop_with_empty_axes", {0}).int_() != 0) {
                return Ok(Tensors{std::move(inputs[0])});
            } else if (keepdims) {
                return Ok(Tensors{std::make_shared<Tensor>(inputs[0]->dataType, Shape(inputs[0]->shape.size(), DimExpr(1)))});
            } else {
                return Ok(Tensors{std::make_shared<Tensor>(inputs[0]->dataType, Shape{})});
            }
        }
    }
}// namespace refactor::onnx
