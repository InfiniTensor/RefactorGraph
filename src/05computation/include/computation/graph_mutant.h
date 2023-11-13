#ifndef COMPUTATION_GRAPH_MUTANT_H
#define COMPUTATION_GRAPH_MUTANT_H

#include "graph_topo.h"
#include "kernel/graph.h"
#include "operator.h"

namespace refactor::computation {
    using kernel::Shape;
    using kernel::Tensor;

    struct Node {
        Arc<MyOperator> op;
        std::string name;
    };

    struct Edge {
        Arc<Tensor> tensor;
        std::string name;
    };

    class GraphMutant {
        graph_topo::LinkedGraph<Node, Edge> _internal;

    public:
        explicit GraphMutant(graph_topo::Graph<Node, Edge>) noexcept;
        GraphMutant(graph_topo::GraphTopo, std::vector<Node>, std::vector<Edge>) noexcept;
        GraphMutant(graph_topo::LinkedGraph<Node, Edge>) noexcept;

        GraphMutant clone() noexcept;

        auto internal() const -> decltype(_internal) const &;
        auto internal() -> decltype(_internal) &;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_MUTANT_H
