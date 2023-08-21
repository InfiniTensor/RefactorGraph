#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include "common/data_type.h"
#include "graph_topo/graph_topo_searcher.hpp"
#include <cstring>
#include <variant>

namespace refactor::graph {
    using dim_t = int64_t;
    using Shape = std::vector<dim_t>;

    /// @brief 非全局输入边填写之前的状态。
    struct EmptyEdgeInfo {
        bool operator==(EmptyEdgeInfo const &) const;
        bool operator!=(EmptyEdgeInfo const &) const;
    };

    /// @brief 张量边。
    struct Tensor {
        common::DataType dataType;
        Shape shape;

        bool operator==(Tensor const &) const;
        bool operator!=(Tensor const &) const;
    };

    /// @brief `shape` 算子产生的边，作为变量，但在边推理中立即产生值。
    struct ShapeVariable {
        Shape shape;

        bool operator==(ShapeVariable const &) const;
        bool operator!=(ShapeVariable const &) const;
    };

    /// @brief 边信息可能是某一种。
    struct EdgeInfo {
        std::variant<EmptyEdgeInfo, Tensor, ShapeVariable> info;

        EdgeInfo();
        EdgeInfo(Tensor);
        EdgeInfo(ShapeVariable);

        bool isEmpty() const;
        bool isTensor() const;
        bool isShapeVariable() const;

        Tensor &tensor();
        Tensor const &tensor() const;
        ShapeVariable &shapeVariable();
        ShapeVariable const &shapeVariable() const;

        bool operator==(EdgeInfo const &) const;
        bool operator!=(EdgeInfo const &) const;
    };

}// namespace refactor::graph

#endif// EDGE_INFO_H
