#include "computation/operators/transpose.h"
#include "common.h"
#include "transpose.hh"
#include <execution>

namespace refactor::onnx {
    using Op = Transpose;

    Op::Transpose(Ints perm_)
        : Operator(), perm(perm_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto perm = defaultOr(attributes, "perm", {}).ints();
        return OpBox(std::make_unique<Op>(std::move(perm)));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Transpose"; }
    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        auto const &data = inputs[0];
        if (!perm.empty()) {
            if (perm.size() != data.shape.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape output(perm.size(), DimExpr(1));
            std::transform(std::execution::unseq,
                           perm.begin(), perm.end(), output.begin(),
                           [&data](auto i) { return data.shape[i]; });
            return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs))});
        } else {
            Shape output(data.shape.rbegin(), data.shape.rend());
            return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs))});
        }
    }
    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Transpose;

        decltype(Op_::perm) perm_(inputs[0].rank());
        if (!perm.empty()) {
            std::transform(std::execution::unseq,
                           perm.begin(), perm.end(), perm_.begin(),
                           [](auto i) { return static_cast<uint32_t>(i); });
        } else {
            std::iota(perm_.rbegin(), perm_.rend(), 0);
        }
        return std::make_unique<Op_>(std::move(perm_));
    }

}// namespace refactor::onnx
