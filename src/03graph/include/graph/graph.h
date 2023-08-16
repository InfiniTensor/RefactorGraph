#ifndef GRAPH_H
#define GRAPH_H

#include "common/op_type.h"
#include "edge_info.h"
#include "graph_topo/graph_topo_searcher.hpp"

namespace refactor::graph {
    struct NodeInfo {
        common::OpType opType;
    };

    class Graph {
        GraphTopoSearcher<NodeInfo, EdgeInfo> topoSearcher;

        void fillEdgeInfo();

    public:
    };

}// namespace refactor::graph

#endif// GRAPH_H
