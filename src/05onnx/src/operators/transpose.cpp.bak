#include "computation/operators/transpose.h"
#include "common.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferTranspose(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1)

        auto const &data = inputs[0];
        if (auto it = op.attributes.find("perm"); it != op.attributes.end()) {
            auto const &perm = it->second.ints();
            if (perm.size() != data.shape.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape output(perm.size(), DimExpr(1));
            std::transform(std::execution::unseq,
                           perm.begin(), perm.end(), output.begin(),
                           [&data](auto i) { return data.shape[i]; });
            return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs))});
        } else {
            Shape output(data.shape.rbegin(), data.shape.rend());
            return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs))});
        }
    }

    LowerOperator lowerTranspose(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        decltype(Transpose::perm) perm_(inputs[0].rank());
        if (auto it = op.attributes.find("perm"); it != op.attributes.end()) {
            auto const &perm = it->second.ints();
            std::transform(std::execution::unseq,
                           perm.begin(), perm.end(), perm_.begin(),
                           [](auto i) { return static_cast<uint32_t>(i); });
        } else {
            std::iota(perm_.rbegin(), perm_.rend(), 0);
        }
        return {std::make_shared<Transpose>(std::move(perm_)), {0}};
    }
}// namespace refactor::onnx
