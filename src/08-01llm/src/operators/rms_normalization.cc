#include "rms_normalization.hh"
#include "common.h"
#include "computation/operators/rms_normalization.h"

namespace refactor::llm {
    using Op = RmsNormalization;

    Op::RmsNormalization(decltype(epsilon) epsilon_)
        : Operator(), epsilon(epsilon_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto epsilon = attributes.getOrInsert("epsilon", {1e-5f}).float_();
        return OpBox(std::make_unique<Op>(epsilon));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::RmsNormalization"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &x = inputs[0];
        auto const &w = inputs[1];
        if (x.rank() < 1 || w.rank() != 1 || x.shape.back() != w.shape.back()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (x.dataType != w.dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        return Ok(Tensors{Tensor::share(x.dataType, x.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::RmsNormalization;
        return std::make_unique<Op_>(epsilon);
    }

}// namespace refactor::llm
