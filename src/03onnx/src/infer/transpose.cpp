#include "infer.h"

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
                Shape ans(perm.size(), DimExpr(1));
                std::transform(perm.begin(), perm.end(), ans.begin(),
                               [&data](auto i) { return data->shape[i]; });
                return Ok(Tensors{Tensor::share(data->dataType, std::move(ans))});
            } else {
                Shape ans(data->shape.rbegin(), data->shape.rend());
                return Ok(Tensors{Tensor::share(data->dataType, std::move(ans))});
            }
        }
    }
}// namespace refactor::onnx
