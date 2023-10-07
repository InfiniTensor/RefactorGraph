﻿#include "simple_binary.hh"
#include "common.h"
#include "common/error_handler.h"
#include "common/range.h"
#include "computation/operators/simple_binary.h"

namespace refactor::onnx {
    using namespace common;
    using Op = SimpleBinary;
    using Ty = SimpleBinaryType;

    Op::SimpleBinary(Ty type_)
        : Operator(), type(type_) {}

    auto Op::build(std::string_view opType, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Simple binary operator should not have attributes");

        if (opType == "onnx::Add") {
            return OpBox(std::make_unique<Op>(Ty::Add));
        }
        if (opType == "onnx::Sub") {
            return OpBox(std::make_unique<Op>(Ty::Sub));
        }
        if (opType == "onnx::Mul") {
            return OpBox(std::make_unique<Op>(Ty::Mul));
        }
        if (opType == "onnx::Div") {
            return OpBox(std::make_unique<Op>(Ty::Div));
        }
        if (opType == "onnx::Pow") {
            return OpBox(std::make_unique<Op>(Ty::Pow));
        }
        UNREACHABLEX(void, "Unsupported binary operator: {}", opType);
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Add: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sub: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Mul: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Div: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Pow: {
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
            case Ty::Add:
                return "onnx::Add";
            case Ty::Sub:
                return "onnx::Sub";
            case Ty::Mul:
                return "onnx::Mul";
            case Ty::Div:
                return "onnx::Div";
            case Ty::Pow:
                return "onnx::Pow";
            default:
                UNREACHABLE();
        }
    }

    template<decltype(DataType::internal) T>
    void calculate(Ty ty, void *dst, void const *a, void const *b) {
        using T_ = typename primitive_t<T>::type;
        auto a_ = *reinterpret_cast<T_ const *>(a);
        auto b_ = *reinterpret_cast<T_ const *>(b);
        auto dst_ = reinterpret_cast<T_ *>(dst);
        switch (ty) {
            case Ty::Add:
                *dst_ = a_ + b_;
                break;
            case Ty::Sub:
                *dst_ = a_ - b_;
                break;
            case Ty::Mul:
                *dst_ = a_ * b_;
                break;
            case Ty::Div:
                *dst_ = a_ / b_;
                break;
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto dataType = a.dataType;
        if (type == Ty::Pow) {
            if (!dataType.isFloat() || !b.dataType.isNumberic()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else {
            if (!dataType.isNumberic() || b.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        }

        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans}) || type == Ty::Pow) {
            return Ok(Tensors{std::move(ans)});
        }

        auto eleSize = dataType.size();
        auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
        for (auto i : range0_(ans->elementsSize())) {
            auto indices = locateN(ans->shape, i);
            auto a_ = locate1(a, indices),
                 b_ = locate1(b, indices);
            auto dst_ = dst + i * eleSize;
            //-------------------------------------
#define CASE(T)                                     \
    case DataType::T:                               \
        calculate<DataType::T>(type, dst_, a_, b_); \
        break
            //-------------------------------------
            switch (dataType.internal) {
                CASE(F32);
                CASE(F64);
                CASE(I32);
                CASE(I64);
                CASE(I8);
                CASE(I16);
                CASE(U8);
                CASE(U16);
                CASE(U32);
                CASE(U64);
                default:
                    ans->free();
                    break;
            }
        }
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> LowerOperator {
        using Op_ = computation::SimpleBinary;
        using Ty_ = computation::SimpleBinaryType;
        Ty_ type_;
        switch (type) {
            case Ty::Add:
                type_ = Ty_::Add;
                break;
            case Ty::Sub:
                type_ = Ty_::Sub;
                break;
            case Ty::Mul:
                type_ = Ty_::Mul;
                break;
            case Ty::Div:
                type_ = Ty_::Div;
                break;
            case Ty::Pow:
                type_ = Ty_::Pow;
                break;
            default:
                UNREACHABLE();
        }
        return {std::make_shared<Op_>(type_), {0, 1}};
    }

}// namespace refactor::onnx
