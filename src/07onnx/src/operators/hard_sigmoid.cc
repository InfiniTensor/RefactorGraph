#include "computation/operators/hard_sigmoid.h"
#include "common.h"
#include "hard_sigmoid.hh"
#include <execution>

namespace refactor::onnx {
    using Op = HardSigmoid;

    Op::HardSigmoid(Float alpha, Float beta)
        : Operator(), alpha(alpha), beta(beta) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto alpha = defaultOr(attributes, "alpha", {0.2f}).float_();
        auto beta = defaultOr(attributes, "beta", {0.5f}).float_();
        return OpBox(std::make_unique<Op>(alpha, beta));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::HardSigmoid"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)
        auto dataType = inputs[0].dataType;
        if (!dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }
        auto ans = Tensor::share(dataType, inputs[0].shape, extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }
    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::HardSigmoid;
        return std::make_unique<Op_>(alpha, beta);
    }


}// namespace refactor::onnx

