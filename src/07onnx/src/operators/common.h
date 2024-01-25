#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include "common.h"
#include "frontend/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace frontend;

    using OptionalInts = std::optional<Ints>;
    using OptionalIntsRef = std::optional<std::reference_wrapper<Ints const>>;

    constexpr Int StandardOpsetVersion = 18;

    /// @brief 池化形状推断。
    /// @param data 输入张量的形状。
    /// @param kernel kernel 的形状。
    /// @param dilations 空洞参数。
    /// @param pads 扩张参数。
    /// @param strides 跳步参数。
    /// @return 池化后的形状。
    ShapeResult pool(SmallInts<4> const &data,
                     Ints const &kernel,
                     OptionalIntsRef const &dilations,
                     OptionalIntsRef const &pads,
                     OptionalIntsRef const &strides);

}// namespace refactor::onnx

#endif// ONNX_INFER_H
