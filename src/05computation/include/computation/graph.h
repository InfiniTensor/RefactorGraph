#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "kernel/graph.h"
#include "operator.h"

namespace refactor::computation {
    using kernel::Shape;
    using kernel::Tensor;

    struct Node {
        OpBox op;
        std::string name;
    };

    struct Edge {
        std::shared_ptr<Tensor> tensor;
        std::string name;
    };

    class Graph {
        graph_topo::PolymorphGraph<Node, Edge> _internal;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge>) noexcept;
        Graph(graph_topo::GraphTopo, std::vector<Node>, std::vector<Edge>) noexcept;

        void transpose();
        kernel::Graph lower(Target);
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_H
