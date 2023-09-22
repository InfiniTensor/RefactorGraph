#include "common.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferUnary(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto dataType = inputs[0]->dataType;
            auto opType = op.opType.name();
            static std::unordered_set<std::string_view> const SET[]{
                {"onnx::Abs", "onnx::Relu", "onnx::PRelu"},
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

    computation::SharedOp lowerUnary(Operator const &op) {
        return nullptr;
    }
}// namespace refactor::onnx
