#include "common/range.h"
#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferExpand(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2)
        if (inputs[1]->dataType != DataType::I64 ||
            inputs[1]->shape.size() != 1 ||
            !inputs[1]->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            auto const &data = inputs[0];
            auto const &shape = inputs[1];
            auto shape_ = reinterpret_cast<int64_t *>(shape->data->ptr);
            EXPECT_VAL(shape->shape[0], shapeSize)
            Shape shape__(shape_, shape_ + shapeSize);

            auto res = multidirBroadcast({data->shape, shape__});
            if (res.isErr()) {
                return Err(InferError(ERROR_MSG(res.unwrapErr())));
            }
            auto dataType = data->dataType;
            auto ans = Tensor::share(dataType, std::move(res.unwrap()));
            if (!shouldCalculate(inputs, ans->shape)) {
                return Ok(Tensors{std::move(ans)});
            }

            auto eleSize = dataTypeSize(dataType);
            auto dst = reinterpret_cast<uint8_t *>(ans->malloc());
            for (auto i : range0_(ans->elementsSize())) {
                auto src = locate1(*data, locateN(ans->shape, i));
                std::memcpy(dst + i * eleSize, src, eleSize);
            }
            return Ok(Tensors{std::move(ans)});
        }
    }
}// namespace refactor::onnx
