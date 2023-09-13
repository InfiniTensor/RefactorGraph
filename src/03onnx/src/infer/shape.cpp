#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferShape(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto attrs = op.attributes;
            auto start = op.attribute("start", {0}).int_(),
                 end = op.attribute("end", {-1}).int_();

            ASSERT(start == 0, "only support start == 0");
            ASSERT(end == -1, "only support end == -1");

            auto ans = inputs[0]->shape;
            auto blob = std::make_shared<Blob>(new int64_t[ans.size()]);
            for (auto i = 0; i < ans.size(); ++i) {
                EXPECT_VAL(ans[i], v)
                reinterpret_cast<int64_t *>(blob->ptr)[i] = v;
            }
            return Ok(Edges{
                std::make_shared<Tensor>(
                    DataType::I64,
                    Shape{DimExpr(ans.size())},
                    std::move(blob)),
            });
        }
    }
}// namespace refactor::onnx
