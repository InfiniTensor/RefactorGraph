#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferPow(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2) {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (!isSignedDataType(a->dataType) || !isNumbericDataType(b->dataType)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            MULTIDIR_BROADCAST((ShapeRefs{a->shape, b->shape}))
            return Ok(Tensors{Tensor::share(a->dataType, std::move(output), extractDependency(inputs))});
        }
    }
}// namespace refactor::onnx
