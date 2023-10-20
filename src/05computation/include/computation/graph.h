#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "graph_topo.h"
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
        Arc<Tensor> tensor;
        std::string name;
    };

    class Graph {
        graph_topo::PolymorphGraph<Node, Edge> _internal;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge>) noexcept;
        Graph(graph_topo::GraphTopo, std::vector<Node>, std::vector<Edge>) noexcept;

        void layoutPermute();

        kernel::Graph lower(Target) const;
        auto internal() const -> decltype(_internal) const &;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_H
