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
        auto dataType = inputs[0].dataType;
        if (!dataType.isNumberic()) {
            return Err(InferError(ERROR_MSG("Data type error")));
        }
        auto notation = computation::EinsteinNotation(equation);
        if (inputs.size() != notation.items()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        std::vector<DimExpr> size(26, DimExpr(1));
        for (auto i = 0; i < inputs.size(); ++i) {
            if (inputs[i].dataType != dataType) {
                return Err(InferError(ERROR_MSG("Data type error")));
            }
            auto indices = notation.indices(i);
            if (inputs[i].rank() != indices.size()) {
                return Err(InferError(ERROR_MSG("Input shape error")));
            }
            for (auto j = 0; j < indices.size(); ++j) {
                size[indices[j] - 'a'] = inputs[i].shape[j];
            }
        }
        auto output_ = notation.outputIndices();
        auto output = Shape(output_.size(), DimExpr(1));
        for (auto i = 0; i < output_.size(); ++i) {
            output[i] = size[output_[i] - 'a'];
        }
        return Ok(Tensors{Tensor::share(dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Einsum;
        return std::make_unique<Op_>(equation);
    }

}// namespace refactor::onnx
