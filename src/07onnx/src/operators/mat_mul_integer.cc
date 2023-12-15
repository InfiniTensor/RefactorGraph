#include "mat_mul_integer.hh"
#include "common.h"
#include "computation/operators/mat_mul.h"
#include <unordered_set>

namespace refactor::onnx {
    using Op = MatMulInteger;

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "MatMulInteger operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::MatMulInteger"; }

    static bool checkZeroPoint(TensorRefs inputs, size_t scalarN) {
        if (inputs.size() <= scalarN + 2) {
            return true;
        }
        auto const &t = inputs[scalarN];
        auto const &zp = inputs[scalarN + 2];
        if (zp.dataType != t.dataType) {
            return false;
        }
        if (zp.rank() == 0) {
            return true;
        }
        switch (t.rank()) {
            case 1:
                return zp.shape == decltype(zp.shape){DimExpr(1)};
            case 2:
                return zp.shape == decltype(zp.shape){t.shape[scalarN]};
            default: {
                auto expect = t.shape;
                expect[expect.size() - 1 - scalarN] = DimExpr(1);
                return zp.shape == expect;
            }
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        static const std::unordered_set<decltype(DataType::internal)>
            SET{DataType::I8, DataType::U8};

        if (inputs.size() < 2 || 4 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size not support")));
        }
        if (!checkZeroPoint(inputs, 0) || !checkZeroPoint(inputs, 1)) {
            return Err(InferError(ERROR_MSG("Input zero point not support")));
        }

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        if (!SET.contains(a.dataType) || !SET.contains(b.dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto sa = a.shape, sb = b.shape;
        switch (sa.size()) {
            case 1:
                sa.emplace(sa.begin(), 1);
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
        return Ok(Tensors{Tensor::share(DataType::I32, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::MatMul;
        return std::make_unique<Op_>(1.0, 1.0, false, false);
    }

}// namespace refactor::onnx
