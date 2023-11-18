#include "unsqueeze.hh"
#include "common.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using Op = Unsqueeze;

    Op::Unsqueeze() : Operator() {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Unsqueeze operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Unsqueeze"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &axes = inputs[1];

        if (axes.dataType != DataType::I64 || axes.shape.size() != 1 || !axes.data) {
            return Err(InferError(ERROR_MSG("Axes not support")));
        }
        auto axes_ = axes.data->get<int64_t>();
        EXPECT_VAL(axes.shape[0], axesSize)
        auto rank = data.rank() + axesSize;
        Shape output(rank, DimExpr(-1));
        for (auto axis : slice(axes_, axesSize)) {
            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || rank < axis) {
                return Err(InferError(ERROR_MSG("Axes out of range")));
            }
            ASSERT(output[axis] == DimExpr(-1), "Axes has duplicate");
            output[axis] = DimExpr(1);
        }
        auto it = data.shape.begin();
        for (auto &out : output) {
            if (out == DimExpr(-1)) {
                out = *it++;
            }
        }
        return Ok(Tensors{Tensor::share(data.dataType,
                                        std::move(output),
                                        extractDependency(inputs),
                                        data.data)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Reshape;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
