#include "common.h"
#include "computation/operators/simple_unary.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace common;

    enum class Ty {
        Abs
    };

    template<decltype(DataType::internal) T>
    void abs(void *dst, void const *src) {
        using T_ = typename primitive_t<T>::type;
        *reinterpret_cast<T_ *>(dst) = std::abs(*reinterpret_cast<T_ const *>(src));
    }

    template<decltype(DataType::internal) T>
    void log(void *dst, void const *src) {
        using T_ = typename primitive_t<T>::type;
        *reinterpret_cast<T_ *>(dst) = std::log(*reinterpret_cast<T_ const *>(src));
    }

    InferResult inferUnary(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(1)

        auto dataType = inputs[0].dataType;
        auto opType = op.opType.name();
        static std::unordered_set<std::string_view> const SET[]{
            {"onnx::Abs", "onnx::Relu", "onnx::Erf"},
            {"onnx::Acos", "onnx::Acosh",
             "onnx::Asin", "onnx::Asinh",
             "onnx::Atan", "onnx::Atanh",
             "onnx::Cos", "onnx::Cosh",
             "onnx::Sin", "onnx::Sinh",
             "onnx::Tan"},
            {"onnx::Tanh", "onnx::Sqrt", "onnx::Sigmoid", "onnx::Log"},
            {"onnx::Neg"},
            {"onnx::Identity"}};
        if (SET[0].find(opType) != SET[0].end()) {
            if (!dataType.isNumberic()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[1].find(opType) != SET[1].end()) {
            if (!dataType.isIeee754()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[2].find(opType) != SET[2].end()) {
            if (!dataType.isFloat()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[3].find(opType) != SET[3].end()) {
            if (!dataType.isSigned()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else if (SET[4].find(opType) != SET[4].end()) {
            // nothing to do
        } else {
            RUNTIME_ERROR(fmt::format("{} not support in unary inference", opType));
        }
        auto ans = Tensor::share(dataType, inputs[0].shape, extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }
        if (opType == "onnx::Identity") {
            ans->data = inputs[0].data;
        } else if (opType == "onnx::Abs") {
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
        } else if (opType == "onnx::Log") {
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
        }
        return Ok(Tensors{std::move(ans)});
    }

    LowerOperator lowerUnary(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::Abs")       ? SimpleUnaryType::Abs
                    : op.opType.is("onnx::Acos")    ? SimpleUnaryType::Acos
                    : op.opType.is("onnx::Acosh")   ? SimpleUnaryType::Acosh
                    : op.opType.is("onnx::Asin")    ? SimpleUnaryType::Asin
                    : op.opType.is("onnx::Asinh")   ? SimpleUnaryType::Asinh
                    : op.opType.is("onnx::Atan")    ? SimpleUnaryType::Atan
                    : op.opType.is("onnx::Atanh")   ? SimpleUnaryType::Atanh
                    : op.opType.is("onnx::Cos")     ? SimpleUnaryType::Cos
                    : op.opType.is("onnx::Cosh")    ? SimpleUnaryType::Cosh
                    : op.opType.is("onnx::Sin")     ? SimpleUnaryType::Sin
                    : op.opType.is("onnx::Sinh")    ? SimpleUnaryType::Sinh
                    : op.opType.is("onnx::Tan")     ? SimpleUnaryType::Tan
                    : op.opType.is("onnx::Tanh")    ? SimpleUnaryType::Tanh
                    : op.opType.is("onnx::Relu")    ? SimpleUnaryType::Relu
                    : op.opType.is("onnx::Sqrt")    ? SimpleUnaryType::Sqrt
                    : op.opType.is("onnx::Sigmoid") ? SimpleUnaryType::Sigmoid
                    : op.opType.is("onnx::Erf")     ? SimpleUnaryType::Erf
                                                    : UNREACHABLEX(SimpleUnaryType,
                                                                   "{} not support",
                                                                   op.opType.name());
        return {std::make_shared<SimpleUnary>(type), {0}};
    }
}// namespace refactor::onnx
