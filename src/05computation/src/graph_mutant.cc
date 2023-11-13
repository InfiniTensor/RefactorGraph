#include "computation/graph_mutant.h"

namespace refactor::computation {

    GraphMutant::GraphMutant(graph_topo::Graph<Node, Edge> internal) noexcept
        : _internal(std::move(internal)) {}
    GraphMutant::GraphMutant(graph_topo::GraphTopo topology,
                             std::vector<Node> nodes,
                             std::vector<Edge> edges) noexcept
        : GraphMutant(graph_topo::Graph<Node, Edge>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    GraphMutant::GraphMutant(graph_topo::LinkedGraph<Node, Edge> internal) noexcept
        : _internal(std::move(internal)) {}
    GraphMutant GraphMutant::clone() noexcept {
        auto internal = this->_internal.clone([](Node const &o) -> Node { return o; },
                                              [](Edge const &e) -> Edge { return e; });
        GraphMutant newGraph(std::move(internal));
        return newGraph;
    }

    auto GraphMutant::internal() const -> decltype(_internal) const & { return _internal; }
    auto GraphMutant::internal() -> decltype(_internal) & { return _internal; }

}// namespace refactor::computation
