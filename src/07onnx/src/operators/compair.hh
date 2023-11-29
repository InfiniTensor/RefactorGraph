#ifndef ONNX_COMPAIR_HH
#define ONNX_COMPAIR_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class CompairType {
        EQ,
        GT,
        GE,
        LT,
        LE,
    };

    struct Compair final : public Operator {
        CompairType type;

        explicit Compair(CompairType);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(CompairType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_COMPAIR_HH
