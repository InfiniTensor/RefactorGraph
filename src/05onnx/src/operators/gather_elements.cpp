#include "computation/operators/gather_elements.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferGatherElements(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &indices = inputs[1];
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto const r = data.rank();
        auto const ri = indices.rank();
        if (r != ri) {
            return Err(InferError(ERROR_MSG("data rank not equal indices rank")));
        }
        if (r < 1) {
            return Err(InferError(ERROR_MSG("data rank not >= 1")));
        }
        auto axis = op.attribute("axis", {0}).int_();
        if (axis < 0) {
            axis += r;
        }
        if (axis < 0 || r <= axis) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto output = indices.shape;
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }

    LowerOperator lowerGatherElements(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto rank = inputs[0].rank();
        auto axis = op.attribute("axis", {0}).int_();
        return {std::make_shared<GatherElements>(axis < 0 ? axis + rank : axis, rank), {0, 1}};
    }
}// namespace refactor::onnx
