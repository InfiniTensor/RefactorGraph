#include "common.h"
#include "common/range.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferSqueeze(Operator const &op, TensorRefs inputs, InferOptions const&) {
        switch (inputs.size()) {
            case 1: {
                auto const &data = inputs[0];
                Shape output;
                for (auto const &dim : data.shape) {
                    EXPECT_VAL(dim, val)
                    if (val != 1) {
                        output.push_back(dim);
                    }
                }
                return Ok(Tensors{Tensor::share(data.dataType,
                                                std::move(output),
                                                extractDependency(inputs),
                                                data.data)});
            }
            case 2: {
                auto const &data = inputs[0];
                auto const &axes = inputs[1];
                if (axes.dataType != DataType::I64 || axes.shape.size() != 1 || !axes.hasData()) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                auto rank = data.rank();
                auto axes_ = reinterpret_cast<int64_t *>(axes.data->ptr);
                EXPECT_VAL(axes.shape[0], axesSize)
                std::unordered_set<int64_t> axes__;
                for (auto ptr = axes_; ptr != axes_ + axesSize; ++ptr) {
                    auto axis = *ptr;
                    if (axis < 0) {
                        axis += rank;
                    }
                    if (axis < 0 || rank <= axis) {
                        return Err(InferError(ERROR_MSG("Axes out of range")));
                    }
                    axes__.insert(axis);
                }
                Shape output;
                for (auto i : range0_(data.shape.size())) {
                    if (axes__.erase(i)) {
                        ASSERT(data.shape[i] == DimExpr(1), "Squeeze error");
                    } else {
                        output.push_back(data.shape[i]);
                    }
                }
                return Ok(Tensors{Tensor::share(data.dataType,
                                                std::move(output),
                                                extractDependency(inputs),
                                                data.data)});
            }
            default:
                return Err(InferError(ERROR_MSG("Squeeze need 1 or 2 inputs")));
        }
    }

    computation::SharedOp lowerSqueeze(Operator const &, TensorRefs inputs) {
        using namespace computation;

        return std::make_shared<Reshape>();
    }
}// namespace refactor::onnx
