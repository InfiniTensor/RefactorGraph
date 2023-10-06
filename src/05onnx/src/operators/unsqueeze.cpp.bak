#include "common.h"
#include "common/range.h"
#include "common/slice.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferUnsqueeze(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &axes = inputs[1];

        if (axes.dataType != DataType::I64 || axes.shape.size() != 1 || !axes.hasData()) {
            return Err(InferError(ERROR_MSG("Axes not support")));
        }
        auto axes_ = axes.data->get<int64_t>();
        EXPECT_VAL(axes.shape[0], axesSize)
        auto rank = data.rank() + axesSize;
        Shape output(rank, DimExpr(-1));
        for (auto axis : slice(axes_, axesSize)) {
            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || rank < axis) {
                return Err(InferError(ERROR_MSG("Axes out of range")));
            }
            ASSERT(output[axis] == DimExpr(-1), "Axes has duplicate");
            output[axis] = DimExpr(1);
        }
        auto it = data.shape.begin();
        for (auto &out : output) {
            if (out == DimExpr(-1)) {
                out = *it++;
            }
        }
        return Ok(Tensors{Tensor::share(data.dataType,
                                        std::move(output),
                                        extractDependency(inputs),
                                        data.data)});
    }

    LowerOperator lowerUnsqueeze(Operator const &, TensorRefs) {
        using namespace computation;

        return {std::make_shared<Reshape>(), {0}};
    }
}// namespace refactor::onnx
