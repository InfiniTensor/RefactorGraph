#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferSqueeze(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto const &data = inputs[0];
            auto const &axes = inputs[1];
            if (axes->dataType != DataType::I64 || axes->shape.size() != 1 || !axes->hasData()) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            auto axes_ = reinterpret_cast<int64_t *>(axes->data->ptr);
            EXPECT_VAL(axes->shape[0], axesSize)
            std::vector<int64_t> axes__(axes_, axes_ + axesSize);
            for (auto &i : axes__) {
                if (i < 0) {
                    i += data->shape.size();
                }
            }
            std::sort(axes__.begin(), axes__.end());
            Shape output;
            auto opType = op.opType.name();
            if (opType == "onnx::Squeeze") {
                auto len = data->shape.size();
                auto itx = data->shape.begin();
                auto ity = axes__.begin();
                output = Shape(len, DimExpr(1));
                for (auto i = 0; i < len; ++i) {
                    if (i != *ity) {
                        output[i] = *itx++;
                    } else {
                        ASSERT(*itx++ == DimExpr(1), "Unsqueeze error");
                        ity++;
                    }
                }
            } else if (opType == "onnx::Unsqueeze") {
                auto len = data->shape.size() + axes__.size();
                auto itx = data->shape.begin();
                auto ity = axes__.begin();
                output = Shape(len, DimExpr(1));
                for (size_t i = 0; i < len; ++i) {
                    if (i != *ity) {
                        output[i] = *itx++;
                    } else {
                        output[i] = DimExpr(1);
                        ity++;
                    }
                }
            } else {
                RUNTIME_ERROR(fmt::format("{} not support in squeeze inference", opType));
            }
            // fmt::println("{} passed its data", opType);
            return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(output), data->data)});
        }
    }
}// namespace refactor::onnx
