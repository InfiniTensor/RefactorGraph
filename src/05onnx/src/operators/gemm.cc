#include "gemm.hh"
#include "common.h"
#include "computation/operators/mat_mul.h"
#include <numeric>

namespace refactor::onnx {
    using namespace common;
    using Op = Gemm;

    Op::Gemm(Float alpha_, Float beta_, bool transA_, bool transB_)
        : Operator(),
          alpha(alpha_),
          beta(beta_),
          transA(transA_),
          transB(transB_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto alpha = defaultOr(attributes, "alpha", {1.0f}).float_();
        auto beta = defaultOr(attributes, "beta", {1.0f}).float_();
        auto transA = defaultOr(attributes, "transA", {0}).int_() != 0;
        auto transB = defaultOr(attributes, "transB", {0}).int_() != 0;
        return OpBox(std::make_unique<Op>(alpha, beta, transA, transB));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Gemm"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
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
        if (!transA) {
            m = a0;
            k = a1;
        } else {
            m = a1;
            k = a0;
        }
        if (!transB) {
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

    auto Op::lower(TensorRefs inputs) const -> LowerOperator {
        using Op_ = computation::MatMul;

        decltype(LowerOperator::inputs) inputs_(inputs.size());
        std::iota(inputs_.begin(), inputs_.end(), 0);
        return {
            std::make_unique<Op_>(alpha, beta, transA, transB),
            std::move(inputs_),
        };
    }
}// namespace refactor::onnx
