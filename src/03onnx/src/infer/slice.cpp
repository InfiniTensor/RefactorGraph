#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferSlice(Operator const &op, Edges inputs) {
        if (inputs.size() < 3 || 5 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &data = inputs[0];
            auto const &starts_ = inputs[1];
            auto const &ends_ = inputs[2];
            auto tint = starts_->dataType;
            if (tint != DataType::I32 && tint != DataType::I64) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            if (ends_->dataType != tint) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            if (starts_->shape.size() != 1) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (ends_->shape != starts_->shape) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            EXPECT_VAL(starts_->shape[0], rank)
            if (data->shape.size() != rank) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (!starts_->hasData() || !ends_->hasData()) {
                return Err(InferError(ERROR_MSG("Starts and ends must be constant")));
            }
            int64_t *starts = reinterpret_cast<int64_t *>(starts_->data->ptr),
                    *ends = reinterpret_cast<int64_t *>(ends_->data->ptr),
                    *axes = nullptr,
                    *steps = nullptr;
            std::vector<int64_t> axes__, steps__;
            if (inputs.size() < 4) {
                axes__.resize(rank);
                axes = axes__.data();
                for (int64_t i = 0; i < rank; ++i) {
                    axes[i] = i;
                }
            } else {
                auto const &axes_ = inputs[3];
                if (axes_->dataType != tint || axes_->shape != starts_->shape) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                if (!axes_->hasData()) {
                    return Err(InferError(ERROR_MSG("Axes must be constant")));
                }
                axes = reinterpret_cast<int64_t *>(axes_->data->ptr);
            }
            if (inputs.size() < 5) {
                steps__.resize(rank, 1);
                steps = steps__.data();
            } else {
                auto const &steps_ = inputs[4];
                if (steps_->dataType != tint || steps_->shape != starts_->shape) {
                    return Err(InferError(ERROR_MSG("Steps not support")));
                }
                if (!steps_->hasData()) {
                    return Err(InferError(ERROR_MSG("Steps must be constant")));
                }
                steps = reinterpret_cast<int64_t *>(steps_->data->ptr);
            }

            Shape ans(rank, DimExpr(1));
            for (size_t i = 0; i < rank; ++i) {
                auto axis = axes[i];
                auto start = starts[i];
                auto end = ends[i];
                auto step = steps[i];
                if (axis < 0) {
                    axis += rank;
                }
                EXPECT_VAL(data->shape[axis], dim)
                if (start < 0) {
                    start += dim;
                }
                if (end < 0) {
                    end += dim;
                }
                if (start < 0 || dim <= start || end < 0 || dim < end) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                if (step > 0) {
                    ans[axis] = DimExpr((end - start + step - 1) / step);
                } else if (step < 0) {
                    ans[axis] = DimExpr((end - start - step + 1) / -step);
                } else {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(ans))});
        }
    }
}// namespace refactor::onnx
