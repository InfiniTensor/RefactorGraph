#ifndef ONNX_TILE_HH
#define ONNX_TILE_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct Tile final : public Operator {

        Tile();

        static OpBox build(std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InputVec valueDependentInputs() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_TILE_HH
