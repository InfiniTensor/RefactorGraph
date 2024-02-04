#include "computation/operators/simple_unary.h"
#include "common.h"
#include "computation/operators/identity.h"
#include "simple_unary.hh"
#include <unordered_set>

namespace refactor::onnx {
    using Op = SimpleUnary;
    using Ty = SimpleUnaryType;

    Op::SimpleUnary(Ty type_)
        : Operator(), type(type_) {}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        EXPECT_NO_ATTRI;

        // clang-format off
        auto type =
        opType == "onnx::Abs"     ? Ty::Abs     :
        opType == "onnx::Acos"    ? Ty::Acos    :
        opType == "onnx::Acosh"   ? Ty::Acosh   :
        opType == "onnx::Asin"    ? Ty::Asin    :
        opType == "onnx::Asinh"   ? Ty::Asinh   :
        opType == "onnx::Atan"    ? Ty::Atan    :
        opType == "onnx::Atanh"   ? Ty::Atanh   :
        opType == "onnx::Cos"     ? Ty::Cos     :
        opType == "onnx::Cosh"    ? Ty::Cosh    :
        opType == "onnx::Sin"     ? Ty::Sin     :
        opType == "onnx::Sinh"    ? Ty::Sinh    :
        opType == "onnx::Tan"     ? Ty::Tan     :
        opType == "onnx::Tanh"    ? Ty::Tanh    :
        opType == "onnx::Relu"    ? Ty::Relu    :
        opType == "onnx::Sqrt"    ? Ty::Sqrt    :
        opType == "onnx::Sigmoid" ? Ty::Sigmoid :
        opType == "onnx::Erf"     ? Ty::Erf     :
        opType == "onnx::Log"     ? Ty::Log     :
        opType == "onnx::Not"     ? Ty::Not     :
        opType == "onnx::Neg"     ? Ty::Neg     :
        opType == "onnx::Identity"? Ty::Identity:
        opType == "onnx::HardSwish" ? Ty::HardSwish :
        opType == "onnx::Exp"     ? Ty::Exp :
        UNREACHABLEX(Ty, "Unsupported unary operator: {}", opType);
        // clang-format on

        return OpBox(std::make_unique<Op>(type));
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Abs: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Acos: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Acosh: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Asin: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Asinh: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Atan: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Atanh: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Cos: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Cosh: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sin: {
                static uint8_t ID = 10;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sinh: {
                static uint8_t ID = 11;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Tan: {
                static uint8_t ID = 12;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Tanh: {
                static uint8_t ID = 13;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Relu: {
                static uint8_t ID = 14;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sqrt: {
                static uint8_t ID = 15;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sigmoid: {
                static uint8_t ID = 16;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Erf: {
                static uint8_t ID = 17;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Log: {
                static uint8_t ID = 18;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Not: {
                static uint8_t ID = 19;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Neg: {
                static uint8_t ID = 20;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Identity: {
                static uint8_t ID = 21;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::HardSwish: {
                static uint8_t ID = 22;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Exp: {
                static uint8_t ID = 23;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }
    auto Op::opTypeName() const -> std::string_view {
        // clang-format off
        switch (type) {
            case Ty::Abs      : return "onnx::Abs";
            case Ty::Acos     : return "onnx::Acos";
            case Ty::Acosh    : return "onnx::Acosh";
            case Ty::Asin     : return "onnx::Asin";
            case Ty::Asinh    : return "onnx::Asinh";
            case Ty::Atan     : return "onnx::Atan";
            case Ty::Atanh    : return "onnx::Atanh";
            case Ty::Cos      : return "onnx::Cos";
            case Ty::Cosh     : return "onnx::Cosh";
            case Ty::Sin      : return "onnx::Sin";
            case Ty::Sinh     : return "onnx::Sinh";
            case Ty::Tan      : return "onnx::Tan";
            case Ty::Tanh     : return "onnx::Tanh";
            case Ty::Relu     : return "onnx::Relu";
            case Ty::Sqrt     : return "onnx::Sqrt";
            case Ty::Sigmoid  : return "onnx::Sigmoid";
            case Ty::Erf      : return "onnx::Erf";
            case Ty::Log      : return "onnx::Log";
            case Ty::Not      : return "onnx::Not";
            case Ty::Neg      : return "onnx::Neg";
            case Ty::Identity : return "onnx::Identity";
            case Ty::HardSwish : return "onnx::HardSwish";
            case Ty::Exp      : return "onnx::Exp";
            default: UNREACHABLE();
        }
        // clang-format on
    }

    template<decltype(DataType::internal) T>
    void abs(void *dst, void const *src) {
        using T_ = typename primitive<T>::type;
        *reinterpret_cast<T_ *>(dst) = std::abs(*reinterpret_cast<T_ const *>(src));
    }

    template<decltype(DataType::internal) T>
    void log(void *dst, void const *src) {
        using T_ = typename primitive<T>::type;
        *reinterpret_cast<T_ *>(dst) = std::log(*reinterpret_cast<T_ const *>(src));
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)

        auto dataType = inputs[0].dataType;
        static std::unordered_set<Ty> const SET[]{
            {Ty::Abs, Ty::Relu, Ty::Erf},
            {Ty::Acos, Ty::Acosh,
             Ty::Asin, Ty::Asinh,
             Ty::Atan, Ty::Atanh,
             Ty::Cos, Ty::Cosh,
             Ty::Sin, Ty::Sinh,
             Ty::Tan, Ty::HardSwish},
            {Ty::Tanh, Ty::Sqrt, Ty::Sigmoid, Ty::Log, Ty::Exp},
            {Ty::Neg},
            {Ty::Identity}};
        if (SET[0].contains(type)) {
            if (!dataType.isNumberic()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[1].contains(type)) {
            if (!dataType.isIeee754()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[2].contains(type)) {
            if (!dataType.isFloat()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[3].contains(type)) {
            if (!dataType.isSigned()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[4].contains(type)) {
            // nothing to do
        } else {
            return Err(InferError(ERROR_MSG(fmt::format("{} not support in unary inference", opTypeName()))));
        }
        auto ans = Tensor::share(dataType, inputs[0].shape, extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }
        if (type == Ty::Identity) {
            ans->data = inputs[0].data;
        } else if (type == Ty::Abs) {
            //-------------------------------------
#define CASE(T)                                                       \
    case DataType::T:                                                 \
        abs<DataType::T>(ans->malloc(), inputs[0].data->get<void>()); \
        break
            //-------------------------------------
            switch (dataType.internal) {
                CASE(F32);
                CASE(F64);
                CASE(I32);
                CASE(I64);
                CASE(I8);
                CASE(I16);
                default:
                    ans->free();
                    break;
            }
#undef CASE
        } else if (type == Ty::Log) {
            //-------------------------------------
#define CASE(T)                                                       \
    case DataType::T:                                                 \
        log<DataType::T>(ans->malloc(), inputs[0].data->get<void>()); \
        break
            //-------------------------------------
            switch (dataType.internal) {
                CASE(F32);
                CASE(F64);
                CASE(I32);
                CASE(I64);
                CASE(I8);
                CASE(I16);
                default:
                    ans->free();
                    break;
            }
#undef CASE
        }
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::SimpleUnary;
        using Ty_ = computation::SimpleUnaryType;
        Ty_ type_;
        // clang-format off
        switch (type) {
            case Ty::Abs      : type_ = Ty_::Abs      ; break;
            case Ty::Acos     : type_ = Ty_::Acos     ; break;
            case Ty::Acosh    : type_ = Ty_::Acosh    ; break;
            case Ty::Asin     : type_ = Ty_::Asin     ; break;
            case Ty::Asinh    : type_ = Ty_::Asinh    ; break;
            case Ty::Atan     : type_ = Ty_::Atan     ; break;
            case Ty::Atanh    : type_ = Ty_::Atanh    ; break;
            case Ty::Cos      : type_ = Ty_::Cos      ; break;
            case Ty::Cosh     : type_ = Ty_::Cosh     ; break;
            case Ty::Sin      : type_ = Ty_::Sin      ; break;
            case Ty::Sinh     : type_ = Ty_::Sinh     ; break;
            case Ty::Tan      : type_ = Ty_::Tan      ; break;
            case Ty::Tanh     : type_ = Ty_::Tanh     ; break;
            case Ty::Relu     : type_ = Ty_::Relu     ; break;
            case Ty::Sqrt     : type_ = Ty_::Sqrt     ; break;
            case Ty::Sigmoid  : type_ = Ty_::Sigmoid  ; break;
            case Ty::Erf      : type_ = Ty_::Erf      ; break;
         // case Ty::Log      : type_ = Ty_::Log      ; break;
            case Ty::Not      : type_ = Ty_::Not      ; break;
            case Ty::Neg      : type_ = Ty_::Neg      ; break;
            case Ty::Identity : return std::make_unique<computation::Identity>();
            case Ty::HardSwish      : type_ = Ty_::HardSwish      ; break;
            case Ty::Exp      : type_ = Ty_::Exp      ; break;
            default: UNREACHABLE();
        }
        // clang-format on
        return std::make_unique<Op_>(type_);
    }

}// namespace refactor::onnx
