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
        GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>> _topo;

    public:
        GraphMut(GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>> &&);

        void fillEdgeInfo();

        GraphTopo<NodeInfo, EdgeInfo> intoGraphTopo();
    };

    class Graph {
        GraphTopoSearcher<NodeInfo, EdgeInfo> _topo;

    public:
        Graph(GraphTopo<NodeInfo, EdgeInfo> &&);
        GraphTopoSearcher<NodeInfo, EdgeInfo> const &topo() const;
    };

}// namespace refactor::graph

#endif// GRAPH_H
