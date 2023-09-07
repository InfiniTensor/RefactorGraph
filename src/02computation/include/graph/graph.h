﻿#ifndef GRAPH_H
#define GRAPH_H

#include "common/op_type.h"
#include "edge_info.h"
#include "graph_topo/graph_topo.h"
#include "node_info.h"

namespace refactor::graph {

    using Edge = std::shared_ptr<Tensor>;

    class Graph {
        graph_topo::Graph<NodeInfo, Edge> _internal;

    public:
        void fillEdgeInfo();
    };

}// namespace refactor::graph

#endif// GRAPH_H