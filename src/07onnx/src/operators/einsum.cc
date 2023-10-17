#include "computation/operators/einsum.h"
#include "common.h"
#include "einsum.hh"
#include "refactor/common.h"
#include <variant>

namespace refactor::onnx {
    using Op = Einsum;

    Op::Einsum(std::string equation_)
        : Operator(), equation(std::move(equation_)) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        return OpBox(std::make_unique<Op>(std::move(attributes.at("equation").string())));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Einsum"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }

        fmt::println("{}", computation::EinsteinNotation(equation).toString());
        TODO("");
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Einsum;
        return std::make_unique<Op_>(equation);
    }

}// namespace refactor::onnx
