#include "computation/operators/split.h"
#include "common.h"
#include "common/range.h"
#include "split.hh"
#include <execution>

namespace refactor::onnx {
    using namespace common;
    using Op = Split;

    Op::Split(Int axis_, Int numOutputs_)
        : Operator(),
          axis(axis_),
          numOutputs(numOutputs_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto axis = defaultOr(attributes, "axis", {0}).int_();
        auto numOutputs = defaultOr(attributes, "num_outputs", {0}).int_();
        return OpBox(std::make_unique<Op>(axis, numOutputs));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Softmax"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty() || inputs.size() > 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &input = inputs[0];
        auto rank = input.rank();
        auto axis_ = axis < 0 ? axis + rank : axis;
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Axis error")));
        }
        EXPECT_VAL(input.shape[axis_], total)
        auto dependencies = extractDependency(inputs);
        if (inputs.size() == 1) {
            Tensors ans(numOutputs, nullptr);
            auto each = total + numOutputs - 1 / numOutputs;
            for (auto i : range0_(numOutputs)) {
                if (total > each) {
                    ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                    ans[i]->shape[axis_] = DimExpr(each);
                } else {
                    ASSERT(i == numOutputs - 1, ERROR_MSG("Split error"));
                    ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                    ans[i]->shape[axis_] = DimExpr(total);
                }
            }
            return Ok(std::move(ans));
        } else {
            auto const &split = inputs[1];
            if (split.dataType != DataType::I64 || split.shape.size() != 1 || !split.hasData()) {
                return Err(InferError(ERROR_MSG("Split not support")));
            }
            EXPECT_VAL(split.shape[0], numOutputs)
            auto split_ = split.data->get<int64_t>();
            auto ans = Tensors(numOutputs, nullptr);
            for (auto i : range0_(numOutputs)) {
                ans[i] = Tensor::share(input.dataType, input.shape, dependencies);
                ans[i]->shape[axis_] = DimExpr(split_[i]);
            }
            return Ok(std::move(ans));
        }
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Split;
        auto rank = inputs[0].rank();
        return std::make_unique<Op_>(axis < 0 ? axis + rank : axis, rank);
    }

}// namespace refactor::onnx
