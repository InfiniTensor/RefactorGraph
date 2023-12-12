#include "scatter_nd.hh"
#include "common.h"
#include "computation/operators/scatter_nd.h"

namespace refactor::onnx {
    using Op = ScatterND;

    auto Op::build(ModelContext const &ctx, std::string_view, Attributes attrs) -> OpBox {
        if (auto it = attrs.find("reduction"); it != attrs.end()) {
            ASSERT(it->second.isString() && it->second.string() == "none",
                   "currently only support `reduction = none`");
        }
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::ScatterND"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(3)

        auto const &data = inputs[0];
        auto r = data.rank();
        if (r < 1) { return Err(InferError(ERROR_MSG("Input `data` rank must be >= 1"))); }

        auto const &indices = inputs[1];
        auto q = indices.rank();
        if (q < 1) { return Err(InferError(ERROR_MSG("Input `indices` rank must be >= 1"))); }

        auto const &updates = inputs[2];
        {
            auto const &k_ = *indices.shape.rbegin();
            EXPECT_VAL(k_, k);
            if (updates.rank() != q + r - k - 1) {
                return Err(InferError(ERROR_MSG("Input `updates` rank mismatch")));
            }
            if (!std::equal(indices.shape.begin(), indices.shape.begin() + q - 1,
                            updates.shape.begin())) {
                return Err(InferError(ERROR_MSG("Input `updates` shape mismatch")));
            }
            if (!std::equal(data.shape.rbegin(), data.shape.rbegin() + r - k,
                            updates.shape.rbegin())) {
                return Err(InferError(ERROR_MSG("Input `updates` shape mismatch")));
            }
        }

        if (indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input `indices` must be int64")));
        }
        if (data.dataType != updates.dataType) {
            return Err(InferError(ERROR_MSG("Input `data` and `updates` must have same data type")));
        }

        return Ok(Tensors{Tensor::share(data.dataType, data.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::ScatterND;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
