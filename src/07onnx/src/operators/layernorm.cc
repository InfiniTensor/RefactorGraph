#include "computation/operators/layernorm.h"
#include "common.h"
#include "layernorm.hh"

namespace refactor::onnx {
    using Op = Layernorm;

    Op::Layernorm(Int axis_, Float epsilon_)
        : Operator(), axis(axis_), epsilon(epsilon_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto axis = attributes["axis"].int_();
        auto epsilon = attributes["epsilon"].float_();
        return OpBox(std::make_unique<Op>(axis, epsilon));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::LayerNormalization"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {

        auto const &x = inputs[0];
        auto const &scale = inputs[1];

        if (!x.dataType.isFloat() ||
            !scale.dataType.isFloat()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        return Ok(Tensors{Tensor::share(x.dataType, x.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::LayerNormalization;
        return std::make_unique<Op_>(epsilon, axis);
    }

}// namespace refactor::onnx