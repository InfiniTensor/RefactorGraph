#ifndef ONNX_PAD_HH
#define ONNX_PAD_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    enum class PadMode {
        Constant,
        Reflect,
        Edge,
        Wrap,
    };

    struct Pad final : public Operator {
        PadMode mode;

        Pad(PadMode);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_PAD_HH
