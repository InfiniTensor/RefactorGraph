#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include "common/error_handler.h"
#include "computation/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace computation;
    using ShapeOrNot = std::optional<Shape>;

#define ERROR_MSG(msg) buildMsg(msg, __FILE__, __LINE__)

    InferResult inferUnary(Operator const &, Tensors);
    InferResult inferArithmetic(Operator const &, Tensors);
    InferResult inferGemm(Operator const &, Tensors);
    InferResult inferMatMul(Operator const &, Tensors);
    InferResult inferReshape(Operator const &, Tensors);
    InferResult inferCumSum(Operator const &, Tensors);
    InferResult inferSlice(Operator const &, Tensors);
    InferResult inferSplit(Operator const &, Tensors);
    InferResult inferShape(Operator const &, Tensors);
    InferResult inferWhere(Operator const &, Tensors);
    InferResult inferSqueeze(Operator const &, Tensors);
    InferResult inferUnsqueeze(Operator const &, Tensors);
    InferResult inferCompair(Operator const &, Tensors);
    InferResult inferSoftmax(Operator const &, Tensors);
    InferResult inferPow(Operator const &, Tensors);
    InferResult inferReduce(Operator const &, Tensors);
    InferResult inferConcat(Operator const &, Tensors);
    InferResult inferGather(Operator const &, Tensors);
    InferResult inferCast(Operator const &, Tensors);
    InferResult inferMax(Operator const &, Tensors);
    InferResult inferTranspose(Operator const &, Tensors);
    InferResult inferExpand(Operator const &, Tensors);
    InferResult inferConstant(Operator const &, Tensors);
    InferResult inferConstantOfShape(Operator const &, Tensors);

    using ShapeResult = Result<Shape, std::string>;
    using ShapeRefs = std::vector<std::reference_wrapper<const Shape>>;

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
    ShapeResult pool(Shape const &data,
                     Shape const &kernel,
                     ShapeOrNot const &dilations,
                     ShapeOrNot const &pads,
                     ShapeOrNot const &strides);

#define EXPECT_SIZE(N)                                         \
    if (inputs.size() != (N)) {                                \
        return Err(InferError(ERROR_MSG("Input size error"))); \
    } else

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

    bool shouldCalculate(Tensors const &inputs, Shape const &output);
    std::unordered_set<DimVariable> extractDependency(Tensors const &inputs);

    using Indices = absl::InlinedVector<int64_t, 4>;
    /// @brief 将标量坐标 `k` 展开到 `shape` 空间。
    Indices locateN(Shape const &shape, size_t k);
    /// @brief 在 `tensor` 中定位空间坐标 `indices` 所指向的元素。
    void *locate1(Tensor const &tensor, Indices const &indices);
}// namespace refactor::onnx

#endif// ONNX_INFER_H
