#include "common.h"
#include "common/range.h"
#include <execution>
#include <unordered_set>

namespace refactor::onnx {
    using namespace refactor::common;

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
        auto rank = data->shape.size();
        EXPECT_VAL(starts_->shape[0], rankParam)
        if (rank < rankParam) {
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
        std::unordered_set<int64_t> axes_set;
        for (auto i : range0_(rank)) { axes_set.insert(i); }
        Shape output(rank, DimExpr(1));
        for (auto i : range0_(rankParam)) {
            auto axis = axes ? axes[i] : i;
            if (axis < 0) {
                axis += rank;
            }
            if (!axes_set.erase(axis)) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            auto start = starts[i];
            auto end = ends[i];
            auto step = steps ? steps[i] : 1;
            EXPECT_VAL(data->shape[axis], dim)
            if (start < 0) {
                start += dim;
            }
            if (end < 0) {
                end += dim;
            }
            if (step > 0) {
                if (start < 0) {
                    start = 0;
                } else if (start > dim) {
                    start = dim;
                }
                if (end < 0) {
                    end = 0;
                } else if (end > dim) {
                    end = dim;
                }
            } else if (step < 0) {
                if (start < 0) {
                    start = 0;
                } else if (start > dim - 1) {
                    start = dim - 1;
                }
                if (end < -1) {
                    end = -1;
                } else if (end > dim - 1) {
                    end = dim - 1;
                }
            } else {
                return Err(InferError(ERROR_MSG("Step not support")));
            }
            if (step > 0) {
                output[axis] = DimExpr((end - start + step - 1) / step);
            } else if (step < 0) {
                output[axis] = DimExpr((end - start - step + 1) / -step);
            } else {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
        }
        for (auto axis : axes_set) {
            output[axis] = data->shape[axis];
        }
        auto ans = Tensor::share(data->dataType, std::move(output), extractDependency(inputs));
        if (!shouldCalculate(inputs, output) || (steps && std::any_of(steps, steps + rankParam, [](auto i) { return i != 1; }))) {
            return Ok(Tensors{std::move(ans)});
        }

        Indices axes_(rank, -1);
        if (axes) {
            for (auto i : range0_(rankParam)) { axes_[axes[i]] = i; }
        }
        std::for_each_n(std::execution::par_unseq, natural_t(0), ans->elementsSize(),
                        [&, rank, axes, starts,
                         eleSize = data->dataType.size(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto i) {
                            auto indices = locateN(output, i);
                            Indices indices_(indices.begin(), indices.end());
                            for (size_t j = 0; j < rank; ++j) {
                                if (axes) {
                                    if (axes_[j] >= 0) {
                                        auto start = starts[axes_[j]];
                                        if (start < 0) {
                                            start += data->shape[j].value();
                                        }
                                        indices_[j] += start;
                                    }
                                } else {
                                    indices_[j] += starts[j];
                                }
                            }
                            std::memcpy(dst + i * eleSize, locate1(*data, indices_), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    computation::SharedOp lowerSlice(Operator const &, Tensors) {
        return nullptr;
    }
}// namespace refactor::onnx
