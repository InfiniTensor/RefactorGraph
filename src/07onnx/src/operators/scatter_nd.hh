#ifndef ONNX_SCATTER_ND_HH
#define ONNX_SCATTER_ND_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct ScatterND final : public Operator {

        ScatterND() = default;

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_SCATTER_ND_HH
