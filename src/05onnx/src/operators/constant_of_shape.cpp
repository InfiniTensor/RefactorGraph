﻿#include "common.h"
#include "common/range.h"
#include "common/slice.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferConstantOfShape(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (input.dataType != DataType::I64 ||
            input.shape.size() != 1 ||
            !input.hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            EXPECT_VAL(input.shape[0], shapeSize)
            Shape output(shapeSize, DimExpr(1));
            auto shape = input.data->get<int64_t>();
            auto slice = slice_t<int64_t>{shape, shape + shapeSize};
            std::transform(std::execution::unseq,
                           slice.begin(), slice.end(), output.begin(),
                           [](auto const d) { return DimExpr(d); });
            auto dependencies = input.depVariables;
            if (auto it = op.attributes.find("value"); it != op.attributes.end()) {
                auto const &value = it->second.tensor();
                ASSERT(value->hasData(), "ConstantOfShape value must have data");
                ASSERT(value->shape == Shape{DimExpr(1)}, "ConstantOfShape value must be scalar");
                auto ans = Tensor::share(value->dataType, std::move(output), std::move(dependencies));
                std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                                [src = value->data->get<uint8_t>(),
                                 dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                                 eleSize = value->dataType.size()](auto const i) {
                                    std::memcpy(dst + i * eleSize, src, eleSize);
                                });
                return Ok(Tensors{std::move(ans)});
            } else {
                auto ans = Tensor::share(DataType::F32, std::move(output), std::move(dependencies));
                std::memset(ans->malloc(), 0, ans->bytesSize());
                return Ok(Tensors{std::move(ans)});
            }
        }
    }
}// namespace refactor::onnx