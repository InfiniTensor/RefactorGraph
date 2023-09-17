#include "common/range.h"
#include "common/slice.h"
#include "infer.h"
#include <execution>

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
            auto slice = slice_t<int64_t>{shape, shape + shapeSize};
            std::transform(slice.begin(), slice.end(), ans.begin(),
                           [&size](auto d) { size *= d; return DimExpr(d); });
            if (auto it = op.attributes.find("value"); it != op.attributes.end()) {
                auto const &value = it->second.tensor();
                ASSERT(value->hasData(), "ConstantOfShape value must have data");
                ASSERT(value->shape == Shape{DimExpr(1)}, "ConstantOfShape value must be scalar");
                auto dataType = value->dataType;
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                auto src = reinterpret_cast<uint8_t *>(value->data->ptr);
                auto dst = reinterpret_cast<uint8_t *>(blob->ptr);
                std::for_each_n(std::execution::par_unseq,
                                natural_t(0), size, [src, dst, eleSize](auto i) {
                                    std::memcpy(dst + i * eleSize, src, eleSize);
                                });
                return Ok(Tensors{Tensor::share(dataType, std::move(ans), std::move(blob))});
            } else {
                auto dataType = DataType::F32;
                auto eleSize = dataTypeSize(dataType);
                size *= eleSize;
                auto blob = std::make_shared<Blob>(new uint8_t[size]);
                std::memset(blob->ptr, 0, size);
                return Ok(Tensors{Tensor::share(dataType, std::move(ans), std::move(blob))});
            }
        }
    }
}// namespace refactor::onnx
