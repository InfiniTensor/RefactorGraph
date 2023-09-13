#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferMax(Operator const &op, Tensors inputs) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            std::vector<Shape> shapes;
            shapes.reserve(inputs.size());
            auto dataType = inputs[0]->dataType;
            for (auto &input : inputs) {
                if (input->dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                shapes.emplace_back(std::move(input->shape));
            }
            auto shape = multidirBroadcast(shapes);
            if (shape.isErr()) {
                return Err(InferError(ERROR_MSG(shape.unwrapErr())));
            } else {
                return Ok(Tensors{std::make_shared<Tensor>(dataType, shape.unwrap())});
            }
        }
    }
}// namespace refactor::onnx
