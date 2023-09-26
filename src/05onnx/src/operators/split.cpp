#include "computation/operators/split.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferSplit(Operator const &op, TensorRefs inputs) {
        if (inputs.empty() || inputs.size() > 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &input = inputs[0];
        auto rank = input.rank();
        auto axis = op.attribute("axis", {0}).int_();
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || rank <= axis) {
            return Err(InferError(ERROR_MSG("Axis error")));
        }
        EXPECT_VAL(input.shape[axis], total)
        auto dependencies = extractDependency(inputs);
        if (inputs.size() == 1) {
            auto numOutputs = op.attribute("num_outputs").int_();
            Tensors ans(numOutputs, nullptr);
            auto each = total + numOutputs - 1 / numOutputs;
            for (auto i : range0_(numOutputs)) {
                if (total > each) {
                    ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                    ans[i]->shape[axis] = DimExpr(each);
                } else {
                    ASSERT(i == numOutputs - 1, ERROR_MSG("Split error"));
                    ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                    ans[i]->shape[axis] = DimExpr(total);
                }
            }
            return Ok(std::move(ans));
        } else {
            auto const &split = inputs[1];
            if (split.dataType != DataType::I64 || split.shape.size() != 1 || !split.hasData()) {
                return Err(InferError(ERROR_MSG("Split not support")));
            }
            EXPECT_VAL(split.shape[0], numOutputs)
            auto split_ = reinterpret_cast<int64_t *>(split.data->ptr);
            auto ans = Tensors(numOutputs, nullptr);
            for (auto i : range0_(numOutputs)) {
                ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                ans[i]->shape[axis] = DimExpr(split_[i]);
            }
            return Ok(std::move(ans));
        }
    }

    computation::SharedOp lowerSplit(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto axis = op.attribute("axis", {-1}).int_();
        if (axis < 0) {
            axis += inputs[0].rank();
        }
        return std::make_shared<Split>(static_cast<size_t>(axis));
    }
}// namespace refactor::onnx
