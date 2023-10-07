#include "computation/operators/mat_mul.h"
#include "common.h"
#include "mat_mul.hh"

namespace refactor::onnx {
    using namespace common;
    using Op = MatMul;

    Op::MatMul() : Operator() {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "MatMul operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::MatMul"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (!dataType.isNumberic() || b.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto sa = a.shape, sb = b.shape;
        switch (sa.size()) {
            case 1:
                sa.insert(sa.begin(), DimExpr(1));
                break;
            case 0:
                return Err(InferError(ERROR_MSG("Input shape not support")));
            default:
                break;
        }
        switch (sb.size()) {
            case 1:
                sb.emplace_back(1);
                break;
            case 0:
                return Err(InferError(ERROR_MSG("Input shape not support")));
            default:
                break;
        }
        auto k = sa.back();
        sa.pop_back();
        auto m = sa.back();
        sa.pop_back();
        auto n = sb.back();
        sb.pop_back();
        if (k != sb.back()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        sb.pop_back();
        MULTIDIR_BROADCAST((ShapeRefs{sa, sb}))
        output.emplace_back(std::move(m));
        output.emplace_back(std::move(n));
        return Ok(Tensors{Tensor::share(dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> LowerOperator {
        using Op_ = computation::MatMul;
        return {std::make_shared<Op_>(1.0, 1.0, false, false), {0, 1}};
    }

}// namespace refactor::onnx
