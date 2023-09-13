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
                Shape ans;
                for (auto i : perm) {
                    ans.emplace_back(data->shape[i]);
                }
                return Ok(Tensors{std::make_shared<Tensor>(data->dataType, std::move(ans))});
            } else {
                Shape ans(data->shape.rbegin(), data->shape.rend());
                return Ok(Tensors{std::make_shared<Tensor>(data->dataType, std::move(ans))});
            }
        }
    }
}// namespace refactor::onnx
