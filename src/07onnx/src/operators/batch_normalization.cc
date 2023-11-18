#include "computation/operators/batch_normalization.h"
#include "batch_normalization.hh"
#include "common.h"
#include <numeric>

namespace refactor::onnx {
    using Op = BatchNormalization;

    Op::BatchNormalization(bool trainingMode_, float epsilon_)
        : Operator(),
          trainingMode(trainingMode_),
          epsilon(epsilon_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto trainingMode = defaultOr(attributes, "training_mode", {0}).int_() != 0;
        auto epsilon = defaultOr(attributes, "epsilon", {1e-5f}).float_();
        return OpBox(std::make_unique<Op>(trainingMode, epsilon));
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
            scale.rank() != 1 ||
            scale.shape[0] != x.shape[1] ||
            bias.shape != scale.shape ||
            mean.shape != scale.shape ||
            var.shape != scale.shape) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }

        return Ok(Tensors{Tensor::share(x.dataType, x.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::BatchNormalization;
        return std::make_unique<Op_>(epsilon);
    }

}// namespace refactor::onnx
