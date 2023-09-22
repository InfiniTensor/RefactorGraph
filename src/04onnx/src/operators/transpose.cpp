#include "common.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferTranspose(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &data = inputs[0];
            auto const &attrs = op.attributes;
            if (auto it = attrs.find("perm"); it != attrs.end()) {
                auto const &perm = it->second.ints();
                if (perm.size() != data->shape.size()) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                Shape output(perm.size(), DimExpr(1));
                std::transform(perm.begin(), perm.end(), output.begin(),
                               [&data](auto i) { return data->shape[i]; });
                return Ok(Tensors{Tensor::share(data->dataType, std::move(output), extractDependency(inputs))});
            } else {
                Shape output(data->shape.rbegin(), data->shape.rend());
                return Ok(Tensors{Tensor::share(data->dataType, std::move(output), extractDependency(inputs))});
            }
        }
    }

    computation::SharedOp lowerTranspose(Operator const &) {
        return nullptr;
    }
}// namespace refactor::onnx
