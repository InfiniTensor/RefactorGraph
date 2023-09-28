#include "common.h"
#include <unordered_set>

namespace refactor::onnx {
    using namespace common;

    InferResult inferLogic(Operator const &op, TensorRefs inputs, InferOptions const &) {
        static std::unordered_set<std::string_view> const SET[]{
            {"onnx::Not"},
            {"onnx::And", "onnx::Or", "onnx::Xor"}};

        auto opType = op.opType.name();
        if (SET[0].find(opType) != SET[0].end()) {
            EXPECT_SIZE(1)
            auto dataType = inputs[0].dataType;
            if (!dataType.isBool()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            return Ok(Tensors{Tensor::share(dataType, inputs[0].shape, extractDependency(inputs))});
        }
        if (SET[1].find(opType) != SET[1].end()) {
            EXPECT_SIZE(2)
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (!a.dataType.isBool() || !b.dataType.isBool()) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
            return Ok(Tensors{Tensor::share(a.dataType, std::move(output), extractDependency(inputs))});
        }
        RUNTIME_ERROR(fmt::format("{} not support in logic inference", opType));
    }

}// namespace refactor::onnx
