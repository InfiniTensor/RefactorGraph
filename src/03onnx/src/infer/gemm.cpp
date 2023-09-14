#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferGemm(Operator const &op, Tensors inputs) {
        if (auto size = inputs.size(); size < 2 || 3 < size) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            auto dataType = a->dataType;
            if (!isNumbericDataType(dataType) || b->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (a->shape.size() != 2 || b->shape.size() != 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }

            EXPECT_VAL(a->shape[0], a0)
            EXPECT_VAL(a->shape[1], a1)
            EXPECT_VAL(b->shape[0], b0)
            EXPECT_VAL(b->shape[1], b1)

            size_t m, n, k;
            if (op.attribute("transA", {0}).int_() == 0) {
                m = a0;
                k = a1;
            } else {
                m = a1;
                k = a0;
            }
            if (op.attribute("transB", {0}).int_() == 0) {
                if (b0 != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b1;
            } else {
                if (b1 != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b0;
            }
            if (inputs.size() == 3) {
                auto c = inputs[2];
                if (c->dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                if (!unidirBroadcast(Shape{DimExpr(m), DimExpr(n)}, c->shape)) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            return Ok(Tensors{Tensor::share(dataType, Shape{DimExpr(m), DimExpr(n)})});
        }
    }
}// namespace refactor::onnx
