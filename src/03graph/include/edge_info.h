#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include "data_type.h"
#include "graph_topo_searcher.hpp"
#include <variant>

namespace refactor::graph {

    struct LayoutDim {
        const char *name;
        len_t size, stride;
    };

    /// @brief 非全局输入边填写之前的状态。
    struct EmptyEdgeInfo {};

    /// @brief 张量边。
    struct Tensor {
        common::DataType dataType;
        std::vector<LayoutDim> layout;
    };

    /// @brief `shape` 算子产生的边，作为变量，但在边推理中立即产生值。
    struct ShapeVariable {
        std::vector<len_t> layout;
    };

    /// @brief 边信息可能是某一种。
    struct EdgeInfo {
        std::variant<EmptyEdgeInfo, Tensor, ShapeVariable> info;

        std::optional<Tensor> optionalTensor() {
            if (std::holds_alternative<Tensor>(info)) {
                return {std::get<Tensor>(info)};
            } else {
                return std::nullopt;
            }
        }

        Tensor &tensor() {
            if (std::holds_alternative<Tensor>(info)) {
                return std::get<Tensor>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }

        std::optional<ShapeVariable> optionalShapeVariable() {
            if (std::holds_alternative<ShapeVariable>(info)) {
                return {std::get<ShapeVariable>(info)};
            } else {
                return std::nullopt;
            }
        }

        ShapeVariable &shapeVariable() {
            if (std::holds_alternative<ShapeVariable>(info)) {
                return std::get<ShapeVariable>(info);
            } else {
                RUNTIME_ERROR("edge type error");
            }
        }
    };

}// namespace refactor::graph

#endif// EDGE_INFO_H
