#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferTile(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(2)

        auto const &input = inputs[0];
        auto const &repeats = inputs[1];

        auto rank = input.rank();
        if (repeats.dataType != DataType::I64 || repeats.shape[0].value() != rank || !repeats.hasData()) {
            return Err(InferError(ERROR_MSG("repeats not support")));
        }
        auto shape_ = repeats.data->get<int64_t>();
        EXPECT_VAL(repeats.shape[0], shapeSize)
        Shape repeatsVec(shape_, shape_ + shapeSize);
        Shape output(rank, DimExpr(1));
        for (auto i : range0_(static_cast<size_t>(rank))) {
            auto inputEle = input.shape[i];
            auto repeatsEle = repeatsVec[i];
            if (!inputEle.isValue() || !repeatsEle.isValue()) {
                return Err(InferError(ERROR_MSG("have unknown variable")));
            }
            output[i] = DimExpr(inputEle.value() * repeatsEle.value());
        }
        auto ans = Tensor::share(input.dataType, std::move(output), extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }

}// namespace refactor::onnx
