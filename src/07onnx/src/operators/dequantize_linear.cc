#include "dequantize_linear.hh"
#include "common.h"
#include "computation/operators/dequantize_linear.h"

namespace refactor::onnx {
    using Op = DequantizeLinear;

    Op::DequantizeLinear(Int axis_) noexcept
        : Operator(), axis(axis_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attrs) -> OpBox {
        auto axis = defaultOr(attrs, "axis", {1}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::DynamicQuantizeLinear"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        switch (inputs.size()) {
            case 2:
            case 3:
                break;
            default:
                return Err(InferError(ERROR_MSG("Input size error")));
        }

        auto const &x = inputs[0];
        auto const &xScale = inputs[1];
        if (xScale.rank() != 0) {
            return Err(InferError(ERROR_MSG("Only support per-tensor quantization currently")));
        }
        if (inputs.size() > 2) {
            auto const &xZeroPoint = inputs[2];
            if (xZeroPoint.dataType != x.dataType || xZeroPoint.shape != xScale.shape) {
                return Err(InferError(ERROR_MSG("x_zero_point info mismatch")));
            }
        }

        return Ok(Tensors{Tensor::share(xScale.dataType, x.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::DequantizeLinear;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
