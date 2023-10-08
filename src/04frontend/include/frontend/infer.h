#ifndef FRONTEND_INFER_H
#define FRONTEND_INFER_H

#include "tensor.h"
#include <result.h>

namespace refactor::frontend {

    class Operator;
    using Edges = std::vector<Edge>;

    struct FatalError {};
    struct UnknownVariable {
        std::string name;
    };
    struct InferError : public std::runtime_error {
        std::variant<FatalError, UnknownVariable> value;

        explicit InferError(std::string);
        explicit InferError(UnknownVariable);
    };

    struct InferOptions {
        bool calculate = true;               // 如果为 false 则不计算所有算子。
        size_t calculationByteThreshold = 64;// 只要输出的字节数少于这个数，无论输入多大都计算。
        size_t bytesDilationThreshold = 2;   // 如果计算使输出比输入膨胀这个倍数则不计算。

        bool shouldCalculate(
            TensorRefs,
            std::vector<std::reference_wrapper<Tensor const>>) const;
    };

    using InferResult = Result<std::vector<Tensor_>, InferError>;
    using InferFn = InferResult (*)(Operator const &, TensorRefs, InferOptions const &);

    std::unordered_set<DimVariable> extractDependency(TensorRefs);

    /// @brief 将标量坐标 `k` 展开到 `shape` 空间。
    SmallInts<4> locateN(Shape const &shape, size_t k);
    /// @brief 在 `tensor` 中定位空间坐标 `indices` 所指向的元素。
    void const *locate1(Tensor const &tensor, SmallInts<4> const &indices);

}// namespace refactor::frontend

#endif// FRONTEND_INFER_H
