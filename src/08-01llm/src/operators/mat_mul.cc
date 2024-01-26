#include "computation/operators/mat_mul.h"
#include "common.h"
#include "mat_mul.hh"

namespace refactor::llm {
    using Op = MatMul;

    Op::MatMul(
        decltype(transA) transA_,
        decltype(transB) transB_)
        : Operator(),
          transA(transA_),
          transB(transB_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto transA = attributes.getOrInsert("transA", {0}).int_() != 0;
        auto transB = attributes.getOrInsert("transB", {0}).int_() != 0;
        return OpBox(std::make_unique<Op>(transA, transB));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::MatMul"; }

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
        DimExpr m(1), n(1), ka(1), kb(1);
        if (!transA) {
            m = sa.rbegin()[1];
            ka = sa.rbegin()[0];
        } else {
            m = sa.rbegin()[0];
            ka = sa.rbegin()[1];
        }
        sa.pop_back();
        sa.pop_back();
        if (!transB) {
            kb = sb.rbegin()[1];
            n = sb.rbegin()[0];
        } else {
            kb = sb.rbegin()[0];
            n = sb.rbegin()[1];
        }
        sb.pop_back();
        sb.pop_back();
        ASSERT(ka == kb, "Input shape not support");
        MULTIDIR_BROADCAST((ShapeRefs{sa, sb}))
        output.emplace_back(std::move(m));
        output.emplace_back(std::move(n));
        return Ok(Tensors{Tensor::share(dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::MatMul;
        return std::make_unique<Op_>(1., 1., transA, transB);
    }

}// namespace refactor::llm
