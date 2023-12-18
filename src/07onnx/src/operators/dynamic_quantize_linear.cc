#include "dynamic_quantize_linear.hh"
#include "common.h"
#include "computation/operators/dynamic_quantize_linear.h"

namespace refactor::onnx {
    using Op = DynamicQuantizeLinear;

    auto Op::build(ModelContext const &, std::string_view, Attributes) -> OpBox {
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::DynamicQuantizeLinear"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &x = inputs[0];
        if (x.dataType != DataType::F32) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto deps = extractDependency(inputs);
        return Ok(Tensors{
            Tensor::share(DataType::U8, x.shape, deps),
            Tensor::share(DataType::F32, {}, deps),
            Tensor::share(DataType::U8, {}, deps),
        });
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::DynamicQuantizeLinear;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
