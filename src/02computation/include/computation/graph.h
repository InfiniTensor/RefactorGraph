#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "operator.h"

namespace refactor::computation {

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge> &&);
        std::unordered_set<std::string> fillEdgeInfo();
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_H
