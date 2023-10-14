#include "computation/operators/compair.h"
#include "common.h"
#include "compair.hh"
#include "refactor/common.h"

namespace refactor::onnx {
    using Op = Compair;
    using Ty = CompairType;

    Op::Compair(CompairType type_)
        : Operator(), type(type_) {}

    auto Op::build(std::string_view opType, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Compair operator should not have attributes");

        if (opType == "onnx::Equal") {
            return OpBox(std::make_unique<Op>(Ty::EQ));
        }
        if (opType == "onnx::Greater") {
            return OpBox(std::make_unique<Op>(Ty::GT));
        }
        if (opType == "onnx::GreaterOrEqual") {
            return OpBox(std::make_unique<Op>(Ty::GE));
        }
        if (opType == "onnx::Less") {
            return OpBox(std::make_unique<Op>(Ty::LT));
        }
        if (opType == "onnx::LessOrEqual") {
            return OpBox(std::make_unique<Op>(Ty::LE));
        }
        UNREACHABLEX(void, "Unsupported compair operator: {}", opType);
    }

    auto Op::typeId(CompairType type) -> size_t {
        switch (type) {
            case Ty::EQ: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::GT: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::GE: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::LT: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::LE: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }
    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::EQ:
                return "onnx::Equal";
            case Ty::GT:
                return "onnx::Greater";
            case Ty::GE:
                return "onnx::GreaterOrEqual";
            case Ty::LT:
                return "onnx::Less";
            case Ty::LE:
                return "onnx::LessOrEqual";
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        if (a.dataType != b.dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        auto ans = Tensor::share(DataType::Bool, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans}) || a.dataType != DataType::I64) {// TODO: support other data type
            return Ok(Tensors{std::move(ans)});
        }

        auto dst = reinterpret_cast<bool *>(ans->malloc());
        for (auto i : range0_(ans->elementsSize())) {
            auto indices = locateN(ans->shape, i);
            auto a_ = *reinterpret_cast<int64_t const *>(locate1(a, indices)),
                 b_ = *reinterpret_cast<int64_t const *>(locate1(b, indices));
            switch (type) {
                case Ty::EQ:
                    dst[i] = a_ == b_;
                    break;
                case Ty::GT:
                    dst[i] = a_ > b_;
                    break;
                case Ty::GE:
                    dst[i] = a_ >= b_;
                    break;
                case Ty::LT:
                    dst[i] = a_ < b_;
                    break;
                case Ty::LE:
                    dst[i] = a_ <= b_;
                    break;
                default:
                    UNREACHABLE();
            }
        }
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Compair;
        using Ty_ = computation::CompairType;
        Ty_ type_;
        switch (type) {
            case Ty::EQ:
                type_ = Ty_::EQ;
                break;
            case Ty::GT:
                type_ = Ty_::GT;
                break;
            case Ty::GE:
                type_ = Ty_::GE;
                break;
            case Ty::LT:
                type_ = Ty_::LT;
                break;
            case Ty::LE:
                type_ = Ty_::LE;
                break;
            default:
                UNREACHABLE();
        }
        return std::make_unique<Op_>(type_);
    }

}// namespace refactor::onnx
