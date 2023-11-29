#ifndef ONNX_GLOBAL_POOL_HH
#define ONNX_GLOBAL_POOL_HH

#include "frontend/operator.h"
#include "pool_type.h"

namespace refactor::onnx {
    using namespace frontend;

    struct GlobalPool final : public Operator {
        PoolType type;

        explicit GlobalPool(PoolType);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId(PoolType);

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_GLOBAL_POOL_HH
