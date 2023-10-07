#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include "common/error_handler.h"
#include "frontend/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace frontend;

    using ShapeResult = Result<Shape, std::string>;
    using ShapeRefs = std::vector<std::reference_wrapper<Shape const>>;
    using OptionalInts = std::optional<std::reference_wrapper<Ints const>>;

    /// @brief 多方向形状广播。
    /// @param inputs 所有输入的形状。
    /// @return 广播后的形状。
    ShapeResult multidirBroadcast(ShapeRefs const &);

    /// @brief 单方向形状广播。
    /// @param target 目标形状。
    /// @param test 测试形状。
    /// @return 测试形状是否可以广播到目标形状。
    bool unidirBroadcast(Shape const &target, Shape const &test);

    /// @brief 池化形状推断。
    /// @param data 输入张量的形状。
    /// @param kernel kernel 的形状。
    /// @param dilations 空洞参数。
    /// @param pads 扩张参数。
    /// @param strides 跳步参数。
    /// @return 池化后的形状。
    ShapeResult pool(SmallInts<4> const &data,
                     Ints const &kernel,
                     OptionalInts const &dilations,
                     OptionalInts const &pads,
                     OptionalInts const &strides);

    Attribute defaultOr(Attributes &attrs,
                        std::string const &name,
                        Attribute defaultValue);

#define EXPECT_SIZE(N)                                         \
    if (inputs.size() != (N)) {                                \
        return Err(InferError(ERROR_MSG("Input size error"))); \
    }

#define EXPECT_VAL(DIM, VAL)                                             \
    int64_t VAL;                                                         \
    if ((DIM).hasValue()) {                                              \
        VAL = (DIM).value();                                             \
    } else {                                                             \
        return Err(InferError(UnknownVariable{(DIM.variable()->name)})); \
    }

#define MULTIDIR_BROADCAST(SHAPES)                              \
    Shape output;                                               \
    {                                                           \
        auto res = multidirBroadcast(SHAPES);                   \
        if (res.isErr()) {                                      \
            return Err(InferError(ERROR_MSG(res.unwrapErr()))); \
        }                                                       \
        output = std::move(res.unwrap());                       \
    }
}// namespace refactor::onnx

#endif// ONNX_INFER_H
