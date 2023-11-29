#include "squeeze.hh"
#include "common.h"
#include "computation/operators/reshape.h"
#include <span>

namespace refactor::onnx {
    using Op = Squeeze;

    Op::Squeeze(decltype(axes) axes_) : Operator(), axes(std::move(axes_)) {}

    auto Op::build(ModelContext const &ctx, std::string_view, Attributes attributes) -> OpBox {
        auto iter = ctx.find("opset_version");
        auto opsetVer = iter != ctx.end() ? iter->second.int_() : StandardOpsetVersion;

        if (opsetVer >= 13) {
            ASSERT(attributes.empty(), "Squeeze operator should not have attributes");
            return OpBox(std::make_unique<Op>(
                std::nullopt));
        } else if (auto it = attributes.find("axes"); it != attributes.end()) {
            return OpBox(std::make_unique<Op>(
                std::make_optional(
                    std::make_optional(
                        std::move(it->second.ints())))));
        } else {
            return OpBox(std::make_unique<Op>(
                std::make_optional<std::optional<Ints>>(
                    std::nullopt)));
        }
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Squeeze"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        std::optional<std::span<int64_t const>> axes_ = std::nullopt;
        if (axes) {
            EXPECT_SIZE(1)
            if (*axes) {
                axes_.emplace(**axes);
            }
        } else if (inputs.size() == 2) {
            auto const &axes__ = inputs[1];
            if (axes__.dataType != DataType::I64 || axes__.rank() != 1 || !axes__.data) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            EXPECT_VAL(axes__.shape[0], axesSize)
            axes_.emplace(std::span(axes__.data->get<int64_t>(), axesSize));
        } else {
            EXPECT_SIZE(1)
        }

        if (auto const &data = inputs[0]; axes_) {
            auto rank = data.rank();
            std::unordered_set<int64_t> axes__;
            for (auto axis : *axes_) {
                if (axis < -rank || rank <= axis) {
                    return Err(InferError(ERROR_MSG("Axes out of range")));
                }
                axes__.insert(axis < 0 ? axis + rank : axis);
            }
            Shape output;
            for (auto i : range0_(data.shape.size())) {
                if (axes__.erase(i)) {
                    ASSERT(data.shape[i] == DimExpr(1), "Squeeze error");
                } else {
                    output.push_back(data.shape[i]);
                }
            }
            return Ok(Tensors{Tensor::share(data.dataType,
                                            std::move(output),
                                            extractDependency(inputs),
                                            data.data)});
        } else {
            Shape output;
            for (auto const &dim : data.shape) {
                EXPECT_VAL(dim, val)
                if (val != 1) {
                    output.push_back(dim);
                }
            }
            return Ok(Tensors{Tensor::share(data.dataType,
                                            std::move(output),
                                            extractDependency(inputs),
                                            data.data)});
        }
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Reshape;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
