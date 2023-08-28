#ifndef GRAPH_H
#define GRAPH_H

#include "common/op_type.h"
#include "edge_info.h"
#include "graph_topo/graph_topo_searcher.hpp"
#include "node_info.h"

namespace refactor::graph {

    template<class T>
    struct Cell {
        mutable T value;
        Cell(T &&value) : value(std::forward<T>(value)) {}
    };

    class GraphMut {
        using seacher_t = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>;
        seacher_t _topo;

    public:
        GraphMut(GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>> &&);
        seacher_t const &topo() const;

        /// @brief 填充边信息。
        void fillEdgeInfo();

        /// @brief 萃取子图。
        /// @param 每个要提取的子图包含的节点。
        /// @return 提取到的子图节点。
        std::vector<seacher_t::Node> extract(std::vector<std::vector<seacher_t::Node>> const &);

        /// @brief 内联子图。
        void reduce();

        GraphTopo<NodeInfo, EdgeInfo> intoGraphTopo();
    };

    class Graph {
        using seacher_t = GraphTopoSearcher<NodeInfo, EdgeInfo>;
        seacher_t _topo;

    public:
        Graph(GraphTopo<NodeInfo, EdgeInfo> &&);
        seacher_t const &topo() const;
    };

}// namespace refactor::graph

#endif// GRAPH_H
