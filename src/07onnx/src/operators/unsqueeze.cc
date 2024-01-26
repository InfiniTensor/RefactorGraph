#include "unsqueeze.hh"
#include "common.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using Op = Unsqueeze;

    Op::Unsqueeze(decltype(axes) axes_) : Operator(), axes(std::move(axes_)) {}

    auto Op::build(ModelContext const &ctx, std::string_view opType, Attributes attributes) -> OpBox {
        auto iter = ctx.find("opset_version");
        auto opsetVer = iter != ctx.end() ? iter->second.int_() : StandardOpsetVersion;

        if (opsetVer >= 13) {
            EXPECT_NO_ATTRI;
            return OpBox(std::make_unique<Op>(std::nullopt));
        } else {
            return OpBox(std::make_unique<Op>(std::make_optional(attributes["axes"].ints())));
        }
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Unsqueeze"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }

        auto const &data = inputs[0];
        std::span<int64_t const> axes_;
        if (axes) {
            axes_ = *axes;
        } else {
            EXPECT_SIZE(2)
            auto const &axes__ = inputs[1];
            if (axes__.dataType != DataType::I64 || axes__.shape.size() != 1 || !axes__.data) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            EXPECT_VAL(axes__.shape[0], axesSize)
            axes_ = std::span(axes__.data->get<int64_t>(), axesSize);
        }

        int64_t rank = data.rank() + axes_.size();
        Shape output(rank, DimExpr(-1));
        for (auto axis : axes_) {
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
