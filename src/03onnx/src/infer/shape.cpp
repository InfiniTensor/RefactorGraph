#include "common/range.h"
#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferShape(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &data = inputs[0];
            auto rank = data->shape.size();

            auto start = op.attribute("start", {0}).int_(),
                 end = op.attribute("end", {static_cast<int64_t>(rank)}).int_();
            if (start < 0) {
                start += rank;
            }
            if (start < 0 || rank <= start) {
                return Err(InferError(ERROR_MSG("start out of range")));
            }
            if (end < 0) {
                end += rank;
            }
            if (end <= start || rank < end) {
                return Err(InferError(ERROR_MSG("end out of range")));
            }

            auto ans = Tensor::share(DataType::I64, Shape{DimExpr(end - start)});
            auto dst = reinterpret_cast<int64_t *>(ans->malloc());
            for (auto i : range(start, end)) {
                EXPECT_VAL(data->shape[i], dim)
                dst[i - start] = dim;
            }
            return Ok(Tensors{std::move(ans)});
        }
    }
}// namespace refactor::onnx
