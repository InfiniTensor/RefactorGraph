#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferMax(Operator const &op, Tensors inputs) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        ShapeRefs shapes;
        shapes.reserve(inputs.size());
        auto dataType = inputs[0]->dataType;
        for (auto &input : inputs) {
            if (input->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            shapes.emplace_back(input->shape);
        }
        auto shape = multidirBroadcast(shapes);
        if (shape.isErr()) {
            return Err(InferError(ERROR_MSG(shape.unwrapErr())));
        } else {
            return Ok(Tensors{Tensor::share(dataType, std::move(shape.unwrap()))});
        }
    }
}// namespace refactor::onnx
