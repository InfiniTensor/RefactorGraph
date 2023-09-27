#include "common.h"
#include "computation/operators/mat_mul.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferGemm(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        if (auto size = inputs.size(); size < 2 || 3 < size) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (!dataType.isNumberic() || b.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        if (a.shape.size() != 2 || b.shape.size() != 2) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }

        EXPECT_VAL(a.shape[0], a0)
        EXPECT_VAL(a.shape[1], a1)
        EXPECT_VAL(b.shape[0], b0)
        EXPECT_VAL(b.shape[1], b1)

        int64_t m, n, k;
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
            auto const &c = inputs[2];
            if (c.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (!unidirBroadcast(Shape{DimExpr(m), DimExpr(n)}, c.shape)) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
        }
        return Ok(Tensors{Tensor::share(dataType, Shape{DimExpr(m), DimExpr(n)}, extractDependency(inputs))});
    }

    computation::SharedOp lowerGemm(Operator const &op, TensorRefs) {
        using namespace computation;

        auto alpha = op.attribute("alpha", {1.0f}).float_();
        auto beta = op.attribute("beta", {1.0f}).float_();
        auto transA = op.attribute("transA", {0}).int_() != 0;
        auto transB = op.attribute("transB", {0}).int_() != 0;
        return std::make_shared<MatMul>(alpha, beta, transA, transB);
    }
}// namespace refactor::onnx
