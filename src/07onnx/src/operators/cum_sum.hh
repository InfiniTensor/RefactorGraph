﻿#ifndef ONNX_CUM_SUM_HH
#define ONNX_CUM_SUM_HH

#include "frontend/operator.h"

namespace refactor::onnx {
    using namespace frontend;

    struct CumSum final : public Operator {
        bool exclusive, reverse;

        CumSum(bool exclusive, bool reverse);

        static OpBox build(ModelContext const &, std::string_view, Attributes);
        static size_t typeId();

        size_t opTypeId() const final;
        std::string_view opTypeName() const final;
        InferResult infer(TensorRefs, InferOptions const &) const final;
        computation::OpBox lower(TensorRefs) const final;
    };

}// namespace refactor::onnx

#endif// ONNX_CUM_SUM_HH
