#ifndef GRAPH_H
#define GRAPH_H

#include "common/op_type.h"
#include "edge_info.h"
#include "graph_topo/graph_topo.h"
#include "node_info.h"

namespace refactor::graph {

    using Node = std::shared_ptr<NodeInfo>;
    using Edge = std::shared_ptr<Tensor>;

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge> &&);
        std::unordered_set<std::string> fillEdgeInfo();
    };

}// namespace refactor::graph

#endif// GRAPH_H
