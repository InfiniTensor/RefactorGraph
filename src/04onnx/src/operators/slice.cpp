#include "computation/operators/slice.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;
    using computation::Dim;

    Result<std::vector<Dim>, InferError> buildDims(
        size_t rank,
        size_t size,
        Shape const &data,
        int64_t const *const starts,
        int64_t const *const ends,
        int64_t const *const axes,
        int64_t const *const steps) {
        std::vector<Dim> dims(rank);
        std::vector<bool> flags(rank, false);
        for (auto i : range0_(size)) {
            auto axis = axes ? axes[i] : i;
            auto step = steps ? steps[i] : 1;
            auto start = starts[i];
            auto end = ends[i];

            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || rank <= axis) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            flags[axis] = true;

            EXPECT_VAL(data[axis], dim)

            if (start < 0) {
                start += dim;
            }
            if (end < 0) {
                end += dim;
            }

            if (step > 0) {
                start = std::clamp(start, 0l, dim);
                end = std::clamp(end, 0l, dim);
                dims[axis] = {start, step, end <= start ? 0 : (end - start + step - 1) / step};
            } else if (step < 0) {
                start = std::clamp(start, 0l, dim - 1);
                end = std::clamp(end, -1l, dim - 1);
                dims[axis] = {end, step, start <= end ? 0 : (start - end - step + 1) / -step};
            } else {
                return Err(InferError(ERROR_MSG("Step not support")));
            }
        }
        if (size < rank) {
            for (auto i : range0_(rank)) {
                if (!flags[i]) {
                    EXPECT_VAL(data[i], dim)
                    dims[i] = {0, 1, dim};
                }
            }
        }
        return Ok(std::move(dims));
    }

    InferResult inferSlice(Operator const &op, Tensors inputs) {
        if (inputs.size() < 3 || 5 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
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
        auto rank = data->rank();
        EXPECT_VAL(starts_->shape[0], size)
        if (rank < size) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (!starts_->hasData() || !ends_->hasData()) {
            return Err(InferError(ERROR_MSG("Starts and ends must be constant")));
        }
        int64_t const *const starts = reinterpret_cast<int64_t *>(starts_->data->ptr);
        int64_t const *const ends = reinterpret_cast<int64_t *>(ends_->data->ptr);
        int64_t const *axes = nullptr,
                      *steps = nullptr;
        if (inputs.size() >= 4) {
            auto const &axes_ = inputs[3];
            if (axes_->dataType != tint || axes_->shape != starts_->shape) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            if (!axes_->hasData()) {
                return Err(InferError(ERROR_MSG("Axes must be constant")));
            }
            axes = reinterpret_cast<int64_t *>(axes_->data->ptr);
        }
        if (inputs.size() == 5) {
            auto const &steps_ = inputs[4];
            if (steps_->dataType != tint || steps_->shape != starts_->shape) {
                return Err(InferError(ERROR_MSG("Steps not support")));
            }
            if (!steps_->hasData()) {
                return Err(InferError(ERROR_MSG("Steps must be constant")));
            }
            steps = reinterpret_cast<int64_t *>(steps_->data->ptr);
        }

        auto res = buildDims(rank, size, data->shape, starts, ends, axes, steps);
        if (res.isErr()) {
            return Err(res.unwrapErr());
        }
        auto dims = std::move(res.unwrap());
        Shape output(dims.size(), DimExpr(1));
        std::transform(std::execution::unseq,
                       dims.begin(), dims.end(), output.begin(),
                       [](auto const &dim) { return DimExpr(dim.number); });

        auto ans = Tensor::share(data->dataType, std::move(output), extractDependency(inputs));
        if (!data->data) { return Ok(Tensors{std::move(ans)}); }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&output, &dims, &data, rank,
                         eleSize = data->dataType.size(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto i) {
                            auto indices = locateN(output, i);
                            Indices indices_(indices.begin(), indices.end());
                            for (auto j : range0_(rank)) {
                                indices_[j] *= dims[j].step;
                                indices_[j] += dims[j].start;
                            }
                            std::memcpy(dst + i * eleSize, locate1(*data, indices_), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    computation::SharedOp lowerSlice(Operator const &, TensorRefs inputs) {
        using namespace computation;

        auto const &data = inputs[0];
        auto const &starts_ = inputs[1];
        auto const &ends_ = inputs[2];

        int64_t const
            *starts = reinterpret_cast<int64_t *>(starts_.data->ptr),
            *ends = reinterpret_cast<int64_t *>(ends_.data->ptr),
            *axes = inputs.size() >= 4 ? reinterpret_cast<int64_t *>(inputs[3].data->ptr) : nullptr,
            *steps = inputs.size() == 5 ? reinterpret_cast<int64_t *>(inputs[4].data->ptr) : nullptr;

        auto rank = data.rank();
        auto size = starts_.shape[0].value();

        return std::make_shared<Slice>(buildDims(rank, size, data.shape, starts, ends, axes, steps).unwrap());
    }
}// namespace refactor::onnx
