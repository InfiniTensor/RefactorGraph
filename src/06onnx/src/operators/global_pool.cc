
#include "computation/operators/global_pool.h"
#include "common.h"
#include "global_pool.hh"

namespace refactor::onnx {
    using namespace common;
    using Op = GlobalPool;
    using Ty = PoolType;

    Op::GlobalPool(Ty type_)
        : Operator(), type(type_) {}

    auto Op::build(std::string_view opType, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Global pool operator should not have attributes");

        if (opType == "onnx::GlobalAveragePool") {
            return OpBox(std::make_unique<Op>(Ty::Average));
        }
        if (opType == "onnx::GlobalLpPool") {
            return OpBox(std::make_unique<Op>(Ty::Lp));
        }
        if (opType == "onnx::GlobalMaxPool") {
            return OpBox(std::make_unique<Op>(Ty::Max));
        }
        UNREACHABLEX(void, "Unsupported global pool operator: {}", opType);
    }
    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Average: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Lp: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Max: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }

    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::Average:
                return "onnx::GlobalAveragePool";
            case Ty::Lp:
                return "onnx::GlobalLpPool";
            case Ty::Max:
                return "onnx::GlobalMaxPool";
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (!input.dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto rank = input.rank();
        if (rank < 2) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        Shape output(rank, DimExpr(1));
        output[0] = input.shape[0];
        output[1] = input.shape[1];
        return Ok(Tensors{Tensor::share(input.dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::GlobalPool;
        using Ty_ = computation::PoolType;

        Ty_ type_;
        switch (type) {
            case Ty::Average:
                type_ = Ty_::Average;
                break;
            case Ty::Lp:
                type_ = Ty_::Lp;
                break;
            case Ty::Max:
                type_ = Ty_::Max;
                break;
            default:
                UNREACHABLE();
        }

        return std::make_unique<Op_>(type_);
    }

}// namespace refactor::onnx
