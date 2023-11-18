#ifndef ONNX_BATCH_NORMALIZATION_HH
#define ONNX_BATCH_NORMALIZATION_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct BatchNormalization final : public Operator {
        bool trainingMode;
        float epsilon;

        BatchNormalization(bool, float);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_BATCH_NORMALIZATION_HH
