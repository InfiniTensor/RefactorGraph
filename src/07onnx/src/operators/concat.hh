#ifndef ONNX_CONCAT_HH
#define ONNX_CONCAT_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Concat final : public Operator {
        Int axis;

        explicit Concat(int64_t);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CONCAT_HH
