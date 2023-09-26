#include "computation/operators/reshape.h"
#include "common.h"
#include "common/range.h"
#include "common/slice.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferReshape(Operator const &op, TensorRefs inputs) {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &shape = inputs[1];
        if (shape.dataType != DataType::I64 || shape.rank() != 1 || !shape.hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        }

        ASSERT(op.attribute("allowzero", {0}).int_() == 0, "Not support allowzero");

        auto shape_ = reinterpret_cast<int64_t *>(shape.data->ptr);
        EXPECT_VAL(shape.shape[0], rank)

        Shape output(rank, DimExpr(1));
        int pos_1 = -1, mulOld = 1, mul = 1;
        auto it = data.shape.begin();
        for (auto i : range0_(static_cast<size_t>(rank))) {
            if (shape_[i] == -1) {
                if (pos_1 != -1) {
                    return Err(InferError(ERROR_MSG("Invalid shape value")));
                }
                pos_1 = i;
                if (it != data.shape.end()) {
                    auto const &d = *it++;
                    EXPECT_VAL(d, v)
                    mulOld *= v;
                }
            } else if (shape_[i] == 0) {
                if (it == data.shape.end()) {
                    return Err(InferError(ERROR_MSG("Invalid shape value")));
                }
                auto const &d = *it++;
                output[i] = d;
            } else {
                output[i] = DimExpr(shape_[i]);
                mul *= shape_[i];

                if (it != data.shape.end()) {
                    auto const &d = *it++;
                    EXPECT_VAL(d, v)
                    mulOld *= v;
                }
            }
        }
        while (it != data.shape.end()) {
            auto const &d = *it++;
            EXPECT_VAL(d, v)
            mulOld *= v;
        }

        if (pos_1 != -1) {
            auto div = std::div(mulOld, mul);
            if (div.rem != 0) {
                return Err(InferError(ERROR_MSG("Invalid shape value")));
            } else {
                output[pos_1] = DimExpr(div.quot);
            }
        } else if (mulOld != mul) {
            return Err(InferError(ERROR_MSG("Invalid shape value")));
        }
        return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs), data.data)});
    }

    computation::SharedOp lowerReshape(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        return std::make_shared<Reshape>();
    }
}// namespace refactor::onnx
