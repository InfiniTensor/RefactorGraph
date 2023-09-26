#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include "common/error_handler.h"
#include "frontend/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace frontend;
    using ShapeOrNot = std::optional<Shape>;

#define ERROR_MSG(msg) buildMsg(msg, __FILE__, __LINE__)
    // clang-format off
    InferResult inferConstant        (Operator const &, TensorRefs);
    InferResult inferConstantOfShape (Operator const &, TensorRefs);
    InferResult inferRange           (Operator const &, TensorRefs);
    InferResult inferShape           (Operator const &, TensorRefs);
    InferResult inferArithmetic      (Operator const &, TensorRefs); computation::SharedOp lowerArithmetic (Operator const &, TensorRefs);
    InferResult inferCast            (Operator const &, TensorRefs); computation::SharedOp lowerCast       (Operator const &, TensorRefs);
    InferResult inferCompair         (Operator const &, TensorRefs); computation::SharedOp lowerCompair    (Operator const &, TensorRefs);
    InferResult inferConcat          (Operator const &, TensorRefs); computation::SharedOp lowerConcat     (Operator const &, TensorRefs);
    InferResult inferCumSum          (Operator const &, TensorRefs); computation::SharedOp lowerCumSum     (Operator const &, TensorRefs);
    InferResult inferExpand          (Operator const &, TensorRefs); computation::SharedOp lowerExpand     (Operator const &, TensorRefs);
    InferResult inferGather          (Operator const &, TensorRefs); computation::SharedOp lowerGather     (Operator const &, TensorRefs);
    InferResult inferGemm            (Operator const &, TensorRefs); computation::SharedOp lowerGemm       (Operator const &, TensorRefs);
    InferResult inferMatMul          (Operator const &, TensorRefs); computation::SharedOp lowerMatMul     (Operator const &, TensorRefs);
    InferResult inferPow             (Operator const &, TensorRefs); computation::SharedOp lowerPow        (Operator const &, TensorRefs);
    InferResult inferReduce          (Operator const &, TensorRefs); computation::SharedOp lowerReduce     (Operator const &, TensorRefs);
    InferResult inferReshape         (Operator const &, TensorRefs); computation::SharedOp lowerReshape    (Operator const &, TensorRefs);
    InferResult inferSelect          (Operator const &, TensorRefs); computation::SharedOp lowerSelect     (Operator const &, TensorRefs);
    InferResult inferSlice           (Operator const &, TensorRefs); computation::SharedOp lowerSlice      (Operator const &, TensorRefs);
    InferResult inferSoftmax         (Operator const &, TensorRefs); computation::SharedOp lowerSoftmax    (Operator const &, TensorRefs);
    InferResult inferSplit           (Operator const &, TensorRefs); computation::SharedOp lowerSplit      (Operator const &, TensorRefs);
    InferResult inferSqueeze         (Operator const &, TensorRefs); computation::SharedOp lowerSqueeze    (Operator const &, TensorRefs);
    InferResult inferTranspose       (Operator const &, TensorRefs); computation::SharedOp lowerTranspose  (Operator const &, TensorRefs);
    InferResult inferUnary           (Operator const &, TensorRefs); computation::SharedOp lowerUnary      (Operator const &, TensorRefs);
    InferResult inferUnsqueeze       (Operator const &, TensorRefs); computation::SharedOp lowerUnsqueeze  (Operator const &, TensorRefs);
    InferResult inferWhere           (Operator const &, TensorRefs); computation::SharedOp lowerWhere      (Operator const &, TensorRefs);

    // clang-format on
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
