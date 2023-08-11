#ifndef GRAPH_H
#define GRAPH_H

#include "edge_info.h"
#include "graph_topo_searcher.hpp"
#include "op_type.h"

namespace refactor::graph {
    struct NodeInfo {
        common::OpType opType;
    };

    class Graph {
        GraphTopo<NodeInfo, EdgeInfo> topo;

        void fillEdgeInfo() {
        }

    public:
    };

}// namespace refactor::graph

#endif// GRAPH_H
