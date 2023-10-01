#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include "common/error_handler.h"
#include "frontend/operator.h"
#include <optional>

namespace refactor::onnx {
    using namespace frontend;

#define ERROR_MSG(msg) buildMsg(msg, __FILE__, __LINE__)
    // clang-format off
    InferResult inferConstant           (Operator const &, TensorRefs, InferOptions const&);
    InferResult inferConstantOfShape    (Operator const &, TensorRefs, InferOptions const&);
    InferResult inferRange              (Operator const &, TensorRefs, InferOptions const&);
    InferResult inferShape              (Operator const &, TensorRefs, InferOptions const&);
    InferResult inferArithmetic         (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerArithmetic         (Operator const &, TensorRefs);
    InferResult inferBatchNormalization (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerBatchNormalization (Operator const &, TensorRefs);
    InferResult inferCast               (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerCast               (Operator const &, TensorRefs);
    InferResult inferCompair            (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerCompair            (Operator const &, TensorRefs);
    InferResult inferConcat             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerConcat             (Operator const &, TensorRefs);
    InferResult inferConv               (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerConv               (Operator const &, TensorRefs);
    InferResult inferCumSum             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerCumSum             (Operator const &, TensorRefs);
    InferResult inferExpand             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerExpand             (Operator const &, TensorRefs);
    InferResult inferGather             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerGather             (Operator const &, TensorRefs);
    InferResult inferGatherElements     (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerGatherElements     (Operator const &, TensorRefs);
    InferResult inferGemm               (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerGemm               (Operator const &, TensorRefs);
    InferResult inferGlobalPool         (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerGlobalPool         (Operator const &, TensorRefs);
    InferResult inferMatMul             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerMatMul             (Operator const &, TensorRefs);
    InferResult inferPool               (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerPool               (Operator const &, TensorRefs);
    InferResult inferPow                (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerPow                (Operator const &, TensorRefs);
    InferResult inferReduce             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerReduce             (Operator const &, TensorRefs);
    InferResult inferReshape            (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerReshape            (Operator const &, TensorRefs);
    InferResult inferSelect             (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerSelect             (Operator const &, TensorRefs);
    InferResult inferSlice              (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerSlice              (Operator const &, TensorRefs);
    InferResult inferSoftmax            (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerSoftmax            (Operator const &, TensorRefs);
    InferResult inferSplit              (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerSplit              (Operator const &, TensorRefs);
    InferResult inferSqueeze            (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerSqueeze            (Operator const &, TensorRefs);
    InferResult inferTile               (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerTile               (Operator const &, TensorRefs);
    InferResult inferTranspose          (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerTranspose          (Operator const &, TensorRefs);
    InferResult inferUnary              (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerUnary              (Operator const &, TensorRefs);
    InferResult inferLogic              (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerLogic              (Operator const &, TensorRefs);
    InferResult inferUnsqueeze          (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerUnsqueeze          (Operator const &, TensorRefs);
    InferResult inferWhere              (Operator const &, TensorRefs, InferOptions const&); LowerOperator lowerWhere              (Operator const &, TensorRefs);

    // clang-format on
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
