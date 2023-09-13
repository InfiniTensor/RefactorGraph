#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferReshape(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2)
        if (auto shape = inputs[1];
            shape->dataType != DataType::I64 ||
            shape->shape.size() != 1 ||
            !shape->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            auto const &input = inputs[0]->shape;
            auto shapeValue = reinterpret_cast<int64_t *>(inputs[1]->data->ptr);
            EXPECT_VAL(inputs[1]->shape[0], rank)
            Shape ans(rank, DimExpr(1));
            auto pos_1 = -1;
            for (auto i = 0; i < ans.size(); ++i) {
                switch (shapeValue[i]) {
                    case -1:
                        if (pos_1 != -1) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        pos_1 = i;
                        break;
                    case 0:
                        if (i >= input.size()) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        ans[i] = input[i];
                        break;
                    default:
                        ans[i] = DimExpr(shapeValue[i]);
                        break;
                }
            }
            size_t old_ = 1, new_ = 1;
            for (auto const &d : input) {
                EXPECT_VAL(d, v)
                old_ *= v;
            }
            for (auto const &d : ans) {
                EXPECT_VAL(d, v)
                new_ *= v;
            }
            if (pos_1 != -1) {
                if (old_ % new_ != 0) {
                    return Err(InferError(ERROR_MSG("Invalid shape variable")));
                } else {
                    ans[pos_1] = DimExpr(old_ / new_);
                }
            } else if (old_ != new_) {
                return Err(InferError(ERROR_MSG("Invalid shape variable")));
            }
            return Ok(Tensors{std::make_shared<Tensor>(inputs[0]->dataType, std::move(ans))});
        }
    }
}// namespace refactor::onnx
