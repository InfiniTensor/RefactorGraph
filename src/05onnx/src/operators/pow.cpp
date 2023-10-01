﻿#include "common.h"
#include "computation/operators/simple_binary.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferPow(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        if (!a.dataType.isSignedLarge() || !b.dataType.isNumberic()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        MULTIDIR_BROADCAST((ShapeRefs{a.shape, b.shape}))
        return Ok(Tensors{Tensor::share(a.dataType, std::move(output), extractDependency(inputs))});
    }

    LowerOperator lowerPow(Operator const &op, TensorRefs) {
        using namespace computation;

        return {std::make_shared<SimpleBinary>(SimpleBinaryType::Pow), {0, 1}};
    }
}// namespace refactor::onnx