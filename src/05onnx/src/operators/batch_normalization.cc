#include "computation/operators/batch_normalization.h"
#include "batch_normalization.hh"
#include "common.h"
#include <numeric>

namespace refactor::onnx {
    using namespace common;
    using Op = BatchNormalization;

    Op::BatchNormalization(bool trainingMode_)
        : Operator(), trainingMode(trainingMode_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto trainingMode = defaultOr(attributes, "training_mode", {0}).int_() != 0;
        return OpBox(std::make_unique<Op>(trainingMode));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Op"; }

    auto Op::infer(
        TensorRefs inputs,
        InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(5)

        auto const &x = inputs[0];
        auto const &scale = inputs[1];
        auto const &bias = inputs[2];
        auto const &mean = inputs[3];
        auto const &var = inputs[4];

        if (!x.dataType.isFloat() ||
            !scale.dataType.isFloat() || bias.dataType != scale.dataType ||
            !mean.dataType.isFloat() || var.dataType != mean.dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        if (x.rank() <= 2 ||
            bias.shape != scale.shape ||
            mean.shape != scale.shape ||
            var.shape != scale.shape) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }

        return Ok(Tensors{Tensor::share(x.dataType, x.shape, extractDependency(inputs))});
    }
    auto Op::lower(TensorRefs inputs) const -> LowerOperator {
        using Op_ = computation::BatchNormalization;
        decltype(LowerOperator::inputs) inputs_(inputs.size());
        std::iota(inputs_.begin(), inputs_.end(), 0);
        return {std::make_unique<Op_>(), std::move(inputs_)};
    }

}// namespace refactor::onnx
