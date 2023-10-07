#include "computation/operators/gather_elements.h"
#include "common.h"
#include "common/range.h"
#include "gather_elements.hh"
#include <execution>

namespace refactor::onnx {
    using namespace common;
    using Op = GatherElements;

    Op::GatherElements(Int axis_)
        : Operator(), axis(axis_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto axis = defaultOr(attributes, "axis", {0}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::GatherElements"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &indices = inputs[1];
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto const rank = data.rank();
        auto const ri = indices.rank();
        if (rank != ri) {
            return Err(InferError(ERROR_MSG("data rank not equal indices rank")));
        }
        if (rank < 1) {
            return Err(InferError(ERROR_MSG("data rank not >= 1")));
        }
        auto axis_ = axis < 0 ? axis + rank : axis;
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto output = indices.shape;
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> LowerOperator {
        using Op_ = computation::GatherElements;
        auto rank = inputs[0].rank();
        return {std::make_shared<Op_>(axis < 0 ? axis + rank : axis, rank), {0, 1}};
    }

}// namespace refactor::onnx
