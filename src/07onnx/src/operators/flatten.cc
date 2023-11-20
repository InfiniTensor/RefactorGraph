#include "flatten.hh"
#include "common.h"
#include "computation/operators/reshape.h"

namespace refactor::onnx {
    using Op = Flatten;

    Op::Flatten(Int axis_) : Operator(), axis(axis_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto axis = defaultOr(attributes, "axis", {1}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Flatten"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &data = inputs[0];
        int64_t axis_ = axis < 0 ? axis + data.rank() : axis;
        dim_t output[]{1, 1};
        for (auto i : range0_(axis_)) {
            EXPECT_VAL(data.shape[i], val)
            output[0] *= val;
        }
        for (auto i : range(axis_, data.rank())) {
            EXPECT_VAL(data.shape[i], val)
            output[1] *= val;
        }
        return Ok(Tensors{Tensor::share(data.dataType,
                                        Shape{DimExpr(output[0]), DimExpr(output[1])},
                                        extractDependency(inputs),
                                        data.data)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Reshape;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
