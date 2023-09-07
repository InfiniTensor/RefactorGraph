﻿#ifndef GRAPH_INFER_H
#define GRAPH_INFER_H

#include "common/error_handler.h"
#include "graph/graph.h"
#include <result.h>

namespace refactor::graph {

    using ShapeOrNot = std::optional<Shape>;
    using Edges = std::vector<Edge>;

    struct InferError : public std::runtime_error {
        explicit InferError(std::string &&msg);
    };
    using InferResult = Result<Edges, InferError>;

#define ERROR_MSG(msg) buildMsg(msg, __FILE__, __LINE__)

    InferResult inferUnary(NodeInfo const &, Edges);
    InferResult inferArithmetic(NodeInfo const &, Edges);
    InferResult inferGemm(NodeInfo const &, Edges);
    InferResult inferMatMul(NodeInfo const &, Edges);
    InferResult inferReshape(NodeInfo const &, Edges);
    InferResult inferCumSum(NodeInfo const &, Edges);
    InferResult inferSlice(NodeInfo const &, Edges);
    InferResult inferShape(NodeInfo const &, Edges);
    InferResult inferWhere(NodeInfo const &, Edges);
    InferResult inferSqueeze(NodeInfo const &, Edges);
    InferResult inferEqual(NodeInfo const &, Edges);
    InferResult inferSoftmax(NodeInfo const &, Edges);
    InferResult inferPow(NodeInfo const &, Edges);
    InferResult inferReduce(NodeInfo const &, Edges);
    InferResult inferConcat(NodeInfo const &, Edges);
    InferResult inferGather(NodeInfo const &, Edges);
    InferResult inferCast(NodeInfo const &, Edges);
    InferResult inferMax(NodeInfo const &, Edges);
    InferResult inferTranspose(NodeInfo const &, Edges);
    InferResult inferExpand(NodeInfo const &, Edges);

    using ShapeResult = Result<Shape, std::string>;

    /// @brief 多方向形状广播。
    /// @param inputs 所有输入的形状。
    /// @return 广播后的形状。
    ShapeResult multidirBroadcast(std::vector<Shape> const &);

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

}// namespace refactor::graph

#endif// GRAPH_INFER_H