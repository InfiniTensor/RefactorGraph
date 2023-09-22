#include "common.h"
#include "common/range.h"
#include <execution>

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

            Shape forRef(shape_, shape_ + shapeSize);
            MULTIDIR_BROADCAST((ShapeRefs{data->shape, forRef}))
            auto ans = Tensor::share(data->dataType, std::move(output), extractDependency(inputs));
            if (!shouldCalculate(inputs, ans->shape)) {
                return Ok(Tensors{std::move(ans)});
            }

            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), ans->elementsSize(),
                            [&data, &ans,
                             dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                             eleSize = data->dataType.size()](auto const i) {
                                std::memcpy(dst + i * eleSize, locate1(*data, locateN(ans->shape, i)), eleSize);
                            });
            return Ok(Tensors{std::move(ans)});
        }
    }

    computation::SharedOp lowerExpand(Operator const &, Tensors) {
        return nullptr;
    }
}// namespace refactor::onnx
