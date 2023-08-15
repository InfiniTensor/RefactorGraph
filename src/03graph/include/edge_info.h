#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include "data_type.h"
#include "graph_topo_searcher.hpp"
#include <cstring>
#include <variant>

namespace refactor::graph {

    /// @brief 非全局输入边填写之前的状态。
    struct EmptyEdgeInfo {
        bool operator==(EmptyEdgeInfo const &rhs) const { return true; }
    };

    /// @brief 张量边。
    struct Tensor {
        common::DataType dataType;
        std::vector<len_t> layout;

        bool operator==(Tensor const &rhs) const { return dataType == rhs.dataType && layout == rhs.layout; }
    };

    /// @brief `shape` 算子产生的边，作为变量，但在边推理中立即产生值。
    struct ShapeVariable {
        std::vector<len_t> layout;

        bool operator==(ShapeVariable const &rhs) const { return layout == rhs.layout; }
    };

    /// @brief 边信息可能是某一种。
    struct EdgeInfo {
        std::variant<EmptyEdgeInfo, Tensor, ShapeVariable> info;

        bool isTensor() const { return std::holds_alternative<Tensor>(info); }
        bool isShapeVariable() const { return std::holds_alternative<ShapeVariable>(info); }

        Tensor &tensor() {
            if (std::holds_alternative<Tensor>(info)) {
                return std::get<Tensor>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }

        Tensor const &tensor() const {
            if (std::holds_alternative<Tensor>(info)) {
                return std::get<Tensor>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }

        ShapeVariable &shapeVariable() {
            if (std::holds_alternative<ShapeVariable>(info)) {
                return std::get<ShapeVariable>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }

        ShapeVariable const &shapeVariable() const {
            if (std::holds_alternative<ShapeVariable>(info)) {
                return std::get<ShapeVariable>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }

        bool operator==(EdgeInfo const &rhs) const {
            if (info.index() != rhs.info.index()) {
                return false;
            } else {
                switch (info.index()) {
                    case 0:
                        return std::get<0>(info) == std::get<0>(rhs.info);
                    case 1:
                        return std::get<1>(info) == std::get<1>(rhs.info);
                    case 2:
                        return std::get<2>(info) == std::get<2>(rhs.info);
                    default:
                        RUNTIME_ERROR("Unreachable");
                }
            }
        }
    };

}// namespace refactor::graph

#endif// EDGE_INFO_H
