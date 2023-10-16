#include "computation/operators/slice.h"
#include "common.h"
#include "refactor/common.h"
#include "slice.hh"
#include <execution>

namespace refactor::onnx {
    using computation::Dim;
    using Op = Slice;

    Op::Slice() : Operator() {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Slice operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Slice"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1, 2, 3, 4}; }

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
            int64_t axis = axes ? axes[i] : i,
                    step = steps ? steps[i] : 1,
                    start = starts[i],
                    end = ends[i];

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

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        if (inputs.size() < 3 || 5 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &data = inputs[0];
        auto const &starts_ = inputs[1];
        auto const &ends_ = inputs[2];
        auto tint = starts_.dataType;
        if (tint != DataType::I32 && tint != DataType::I64) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }
        if (ends_.dataType != tint) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }
        if (starts_.shape.size() != 1) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (ends_.shape != starts_.shape) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto rank = data.rank();
        EXPECT_VAL(starts_.shape[0], size)
        if (rank < size) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (!starts_.data || !ends_.data) {
            return Err(InferError(ERROR_MSG("Starts and ends must be constant")));
        }
        int64_t const *starts = starts_.data->get<int64_t>(),
                      *ends = ends_.data->get<int64_t>();
        int64_t const *axes = nullptr,
                      *steps = nullptr;
        if (inputs.size() >= 4) {
            auto const &axes_ = inputs[3];
            if (axes_.dataType != tint || axes_.shape != starts_.shape) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            if (!axes_.data) {
                return Err(InferError(ERROR_MSG("Axes must be constant")));
            }
            axes = axes_.data->get<int64_t>();
        }
        if (inputs.size() == 5) {
            auto const &steps_ = inputs[4];
            if (steps_.dataType != tint || steps_.shape != starts_.shape) {
                return Err(InferError(ERROR_MSG("Steps not support")));
            }
            if (!steps_.data) {
                return Err(InferError(ERROR_MSG("Steps must be constant")));
            }
            steps = steps_.data->get<int64_t>();
        }

        auto res = buildDims(rank, size, data.shape, starts, ends, axes, steps);
        if (res.isErr()) {
            return Err(res.unwrapErr());
        }
        auto dims = std::move(res.unwrap());
        Shape output(dims.size(), DimExpr(1));
        std::transform(std::execution::unseq,
                       dims.begin(), dims.end(), output.begin(),
                       [](auto const &dim) { return DimExpr(dim.number); });

        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        if (!data.data) { return Ok(Tensors{std::move(ans)}); }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&output, &dims, &data, rank,
                         eleSize = data.dataType.size(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto i) {
                            auto indices = locateN(output, i);
                            for (auto j : range0_(rank)) {
                                indices[j] *= dims[j].step;
                                indices[j] += dims[j].start;
                            }
                            std::memcpy(dst + i * eleSize, locate1(data, indices), eleSize);
                        });
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Slice;

        auto const &data = inputs[0];
        auto const &starts_ = inputs[1];
        auto const &ends_ = inputs[2];

        int64_t const
            *starts = starts_.data->get<int64_t>(),
            *ends = ends_.data->get<int64_t>(),
            *axes = inputs.size() >= 4 ? inputs[3].data->get<int64_t>() : nullptr,
            *steps = inputs.size() == 5 ? inputs[4].data->get<int64_t>() : nullptr;

        auto rank = data.rank();
        auto size = starts_.shape[0].value();

        return std::make_unique<Op_>(buildDims(rank, size, data.shape, starts, ends, axes, steps).unwrap());
    }
}// namespace refactor::onnx
