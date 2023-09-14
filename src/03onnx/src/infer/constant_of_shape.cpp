#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferConstantOfShape(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1)
        if (auto input = inputs[0];
            input->dataType != DataType::I64 ||
            input->shape.size() != 1 ||
            !input->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            EXPECT_VAL(input->shape[0], shapeSize)
            Shape ans(shapeSize, DimExpr(1));
            size_t size = 1;
            auto shape = reinterpret_cast<int64_t *>(input->data->ptr);
            for (auto i = 0; i < shapeSize; ++i) {
                auto d = shape[i];
                ans[i] = DimExpr(d);
                size *= d;
            }
            if (auto it = op.attributes.find("value"); it != op.attributes.end()) {
                auto const &value = it->second.tensor();
                ASSERT(value->hasData(), "ConstantOfShape value must have data");
                ASSERT(value->shape.size() == 1 && value->shape[0] == DimExpr(1), "ConstantOfShape value must be scalar");
                auto dataType = value->dataType;
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                for (auto i = 0; i < size; ++i) {
                    std::memcpy(reinterpret_cast<uint8_t *>(blob->ptr) + i * eleSize, value->data->ptr, eleSize);
                }
                return Ok(Tensors{Tensor::share(dataType, std::move(ans), std::move(blob))});
            } else {
                auto dataType = DataType::F32;
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                std::memset(blob->ptr, 0, size * eleSize);
                return Ok(Tensors{Tensor::share(dataType, std::move(ans), std::move(blob))});
            }
        }
    }
}// namespace refactor::onnx
