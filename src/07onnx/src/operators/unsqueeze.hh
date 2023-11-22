#ifndef ONNX_UNSQUEEZE_HH
#define ONNX_UNSQUEEZE_HH

#include "frontend/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace frontend;

    struct Unsqueeze final : public Operator {
        std::optional<Ints> axes;

        explicit Unsqueeze(decltype(axes));

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_UNSQUEEZE_HH
