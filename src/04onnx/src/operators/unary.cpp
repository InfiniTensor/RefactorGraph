#include "common.h"
#include "computation/operators/simple_unary.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferUnary(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto dataType = inputs[0]->dataType;
            auto opType = op.opType.name();
            static std::unordered_set<std::string_view> const SET[]{
                {"onnx::Abs", "onnx::Relu"},
                {"onnx::Acos", "onnx::Acosh",
                 "onnx::Asin", "onnx::Asinh",
                 "onnx::Atan", "onnx::Atanh",
                 "onnx::Cos", "onnx::Cosh",
                 "onnx::Sin", "onnx::Sinh",
                 "onnx::Tan"},
                {"onnx::Tanh", "onnx::Sqrt"}};
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
            } else {
                RUNTIME_ERROR(fmt::format("{} not support in unary inference", opType));
            }
            return Ok(Tensors{Tensor::share(dataType, inputs[0]->shape, extractDependency(inputs))});
        }
    }

    static computation::SimpleUnaryType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in unary lowering", opType.name()));
    }

    computation::SharedOp lowerUnary(Operator const &op, TensorRefs) {
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
                                                    : unsupport(op.opType);
        return std::make_shared<SimpleUnary>(type);
    }
}// namespace refactor::onnx
