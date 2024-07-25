#include "leaky_relu.hh"
#include "common.h"
#include "computation/operators/leaky_relu.h"
#include <execution>

namespace refactor::onnx {
    using Op = LeakyRelu;

    Op::LeakyRelu(Float alpha)
        : Operator(), alpha(alpha) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto alpha = attributes.getOrInsert("alpha", {0.01f}).float_();
        return OpBox(std::make_unique<Op>(alpha));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::LeakyRelu"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)
        auto dataType = inputs[0].dataType;
        if (!dataType.isFloat()) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }
        auto ans = Tensor::share(dataType, inputs[0].shape, extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }
    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::LeakyRelu;
        return std::make_unique<Op_>(alpha);
    }


}// namespace refactor::onnx
