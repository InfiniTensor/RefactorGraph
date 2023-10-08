#include "squeeze.hh"
#include "common.h"
#include "common/range.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using namespace common;
    using Op = Squeeze;

    Op::Squeeze() : Operator() {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        ASSERT(attributes.empty(), "Squeeze operator should not have attributes");
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Squeeze"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        switch (inputs.size()) {
            case 1: {
                auto const &data = inputs[0];
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
            case 2: {
                auto const &data = inputs[0];
                auto const &axes = inputs[1];
                if (axes.dataType != DataType::I64 || axes.shape.size() != 1 || !axes.hasData()) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                auto rank = data.rank();
                auto axes_ = axes.data->get<int64_t>();
                EXPECT_VAL(axes.shape[0], axesSize)
                std::unordered_set<int64_t> axes__;
                for (auto ptr = axes_; ptr != axes_ + axesSize; ++ptr) {
                    auto axis = *ptr;
                    if (axis < 0) {
                        axis += rank;
                    }
                    if (axis < 0 || rank <= axis) {
                        return Err(InferError(ERROR_MSG("Axes out of range")));
                    }
                    axes__.insert(axis);
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
            }
            default:
                return Err(InferError(ERROR_MSG("Squeeze need 1 or 2 inputs")));
        }
    }

    auto Op::lower(TensorRefs) const -> LowerOperator {
        using Op_ = computation::Reshape;
        return {std::make_unique<Op_>(), {0}};
    }

}// namespace refactor::onnx
