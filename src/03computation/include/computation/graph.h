#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "operator.h"
#include "tensor.h"

namespace refactor::computation {

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge>);
        Graph(graph_topo::GraphTopo, std::vector<Node>, std::vector<Edge>);
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_H
