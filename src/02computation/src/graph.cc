#include "computation/graph.h"

namespace refactor::computation {

    Graph::Graph(graph_topo::Graph<Node, Edge> internal) : _internal(std::move(internal)) {}
    Graph::Graph(graph_topo::GraphTopo topology, std::vector<Node> nodes, std::vector<Edge> edges)
        : Graph(graph_topo::Graph<Node, Edge>{std::move(topology)}) {}

}// namespace refactor::computation
