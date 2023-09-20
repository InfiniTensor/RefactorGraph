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
        auto res = multidirBroadcast(shapes);
        if (res.isErr()) {
            return Err(InferError(ERROR_MSG(res.unwrapErr())));
        }
        auto ans = Tensor::share(dataType, std::move(res.unwrap()));
        return Ok(Tensors{std::move(ans)});
    }
}// namespace refactor::onnx
