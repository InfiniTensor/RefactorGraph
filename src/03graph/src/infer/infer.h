#ifndef INFER_H
#define INFER_H

#include "graph/edge_info.h"
#include <result.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace refactor::graph {

    using Edges = std::vector<EdgeInfo>;
    struct InferError : public std::runtime_error {
        explicit InferError(std::string &&msg);
    };
    using InferResult = Result<Edges, InferError>;

#define INFER_ERROR(msg) InferError(buildMsg(msg, __FILE__, __LINE__))

    InferResult inferUnary(Edges, bool(DataType));
    InferResult inferArithmetic(Edges);
    InferResult inferGemm(Edges, bool, bool);
    InferResult inferConv(Edges);
    InferResult inferPool(Edges);
    InferResult inferGlobalPool(Edges);
    InferResult inferReshape(Edges);
    InferResult inferBatchNormalization(Edges);

    using BroadcastResult = Result<Shape, std::string>;
#define BROADCAST_ERROR(msg) buildMsg(msg, __FILE__, __LINE__)

    /// @brief 多方向形状广播。
    /// @param inputs 所有输入的形状。
    /// @return 广播后的形状。
    BroadcastResult multidirBroadcast(std::vector<Shape> const &);

    /// @brief 单方向形状广播。
    /// @param target 目标形状。
    /// @param test 测试形状。
    /// @return 测试形状是否可以广播到目标形状。
    bool unidirBroadcast(Shape target, Shape test);

}// namespace refactor::graph

#endif// INFER_H
