#include "computation/operators/where.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferWhere(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(3)

        auto const &condition = inputs[0];
        auto const &x = inputs[1];
        auto const &y = inputs[2];
        if (condition.dataType != DataType::Bool) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        if (x.dataType != y.dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        MULTIDIR_BROADCAST((ShapeRefs{condition.shape, x.shape, y.shape}))
        auto ans = Tensor::share(x.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&condition, &x, &y, &ans,
                         eleSize = x.dataType.size(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto const i) {
                            auto indices = locateN(ans->shape, i);
                            auto const &selected = *reinterpret_cast<bool const *>(locate1(condition, indices))
                                                       ? x
                                                       : y;
                            std::memcpy(dst + i * eleSize, locate1(selected, indices), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    computation::SharedOp lowerWhere(Operator const &, TensorRefs) {
        using namespace computation;

        return std::make_shared<Where>();
    }
}// namespace refactor::onnx
