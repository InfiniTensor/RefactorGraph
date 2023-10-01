﻿#include "common.h"

namespace refactor::onnx {
    using namespace common;

    template<class T>
    InferResult calculate(void const *start,
                          void const *limit,
                          void const *delta,
                          std::unordered_set<DimVariable> depVariables) {
        auto start_ = *reinterpret_cast<T const *>(start);
        auto limit_ = *reinterpret_cast<T const *>(limit);
        auto delta_ = *reinterpret_cast<T const *>(delta);
        size_t size;
        if constexpr (std::is_floating_point_v<T>) {
            size = std::ceil((limit_ - start_) / delta_);
        } else {
            size = (limit_ - start_) / delta_;
        }
        auto ans = Tensor::share(dataType<T>(), Shape{DimExpr(size)}, std::move(depVariables));
        auto dst = reinterpret_cast<T *>(ans->malloc());
        while (start_ < limit_) {
            *dst++ = start_;
            start_ += delta_;
        }
        return Ok(Tensors{std::move(ans)});
    }

    InferResult inferRange(Operator const &, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(3)

        auto const &start = inputs[0];
        auto const &limit = inputs[1];
        auto const &delta = inputs[2];
        auto dataType = start.dataType;
        if (limit.dataType != dataType || delta.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        }
        if (start.shape.size() != 0 ||
            limit.shape.size() != 0 ||
            delta.shape.size() != 0) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (!start.hasData() ||
            !limit.hasData() ||
            !delta.hasData()) {
            return Err(InferError(ERROR_MSG("Input data not support")));
        }
        //-------------------------------------
#define CASE(T)                                           \
    case DataType::T:                                     \
        return calculate<primitive_t<DataType::T>::type>( \
            start.data->get<void>(),                      \
            limit.data->get<void>(),                      \
            delta.data->get<void>(),                      \
            extractDependency(inputs))
        //-------------------------------------
        switch (dataType.internal) {
            CASE(F32);
            CASE(F64);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::onnx