﻿#include "constant_of_shape.hh"
#include "common.h"
#include <execution>

namespace refactor::onnx {
    using Op = ConstantOfShape;

    Op::ConstantOfShape(Tensor_ value_)
        : Operator(), value(std::move(value_)) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto it = attributes.get("value");
        auto value = it ? std::move(it->get().tensor()) : nullptr;
        return OpBox(std::make_unique<Op>(std::move(value)));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::ConstantOfShape"; }
    auto Op::valueDependentInputs() const -> InputVec { return {0}; }
    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (input.dataType != DataType::I64 ||
            input.shape.size() != 1 ||
            !input.data) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        }

        EXPECT_VAL(input.shape[0], shapeSize)
        Shape output(shapeSize, DimExpr(1));
        std::span slice(input.data->get<int64_t>(), shapeSize);
        std::transform(std::execution::unseq,
                       slice.begin(), slice.end(), output.begin(),
                       [](auto const d) { return DimExpr(d); });
        auto dependencies = input.depVariables;
        if (value) {
            ASSERT(value->data, "ConstantOfShape value must have data");
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
}// namespace refactor::onnx
