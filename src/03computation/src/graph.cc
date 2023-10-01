#include "computation/graph.h"

namespace refactor::computation {

    Graph::Graph(graph_topo::Graph<Node, Edge> internal)
        : _internal(std::move(internal)) {}
    Graph::Graph(graph_topo::GraphTopo topology,
                 std::vector<Node> nodes,
                 std::vector<Edge> edges)
        : Graph(graph_topo::Graph<Node, Edge>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    void Graph::Transpose() {
        fmt::println("Transpose");
        for (auto const &node : _internal.nodes) {
            if (node.op->isLayoutDependent()) {
                fmt::println("Layout dependent op detected: {}({})", node.name, node.op->name());
            }
        }
    }

}// namespace refactor::computation
