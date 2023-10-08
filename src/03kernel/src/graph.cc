#include "kernel/graph.h"

namespace refactor::kernel {

    Graph::Graph(_G internal) noexcept
        : _internal(std::move(internal)) {}
    Graph::Graph(graph_topo::GraphTopo topology,
                 std::vector<_N> nodes,
                 std::vector<_E> edges) noexcept
        : Graph(_G{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

}// namespace refactor::kernel
