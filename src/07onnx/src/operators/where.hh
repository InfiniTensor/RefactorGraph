#ifndef ONNX_WHERE_HH
#define ONNX_WHERE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Where final : public Operator {

        Where();

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_WHERE_HH
