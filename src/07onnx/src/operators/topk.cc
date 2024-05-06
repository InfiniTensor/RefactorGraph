#include "common.h"
#include "topk.hh"
#include "computation/operators/topk.h"
#include <execution>

namespace refactor::onnx {
    using Op = TopK;

    Op::TopK(Int topk, Int axis):Operator(), topk(topk), axis(axis){}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        auto axis = attributes["axis"].int_();
        auto topk = attributes["topk"].int_();
        return OpBox(std::make_unique<Op>(topk, axis));
    }

    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "TopK"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
            if (inputs.empty() || inputs.size() >= 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &input = inputs[0];
        auto rank = input.rank();
        auto axis_ = axis < 0 ? axis + rank : axis;
        if (rank <= axis_) {
            return Err(InferError(ERROR_MSG("axis error")));
        }
        if (topk < 0 || topk > input.shape[axis_].value()){
            return Err(InferError(ERROR_MSG("topk error")));
        }

        Tensors ans(2, nullptr);
        auto dependencies = extractDependency(inputs);
        ans[0] = Tensor::share(input.dataType, input.shape, dependencies);
        ans[0]->shape[axis_] = DimExpr(topk);
        ans[1] = Tensor::share(DataType::U32, input.shape, dependencies);            
        ans[1]->shape[axis_] = DimExpr(topk);  
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::TopK;
        auto rank = inputs[0].rank();
        auto axis_ = axis < 0 ? axis + rank : axis;
        return std::make_unique<Op_>(topk, axis_);
    }

}// namespace refactor::onnx
