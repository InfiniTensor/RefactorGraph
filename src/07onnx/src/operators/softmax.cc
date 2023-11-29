#include "computation/operators/softmax.h"
#include "common.h"
#include "softmax.hh"

namespace refactor::onnx {
    using Op = Softmax;

    Op::Softmax(Int axis_)
        : Operator(), axis(axis_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto axis = defaultOr(attributes, "axis", {-1}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Softmax"; }
    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)
        if (!inputs[0].dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        return Ok(Tensors{Tensor::share(inputs[0])});
    }
    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Softmax;
        auto rank = inputs[0].rank();
        return std::make_unique<Op_>(axis < 0 ? axis + rank : axis, rank);
    }

}// namespace refactor::onnx
