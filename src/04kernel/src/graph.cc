#include "kernel/graph.h"

namespace refactor::kernel {

    Graph::Graph(decltype(_device) device,
                 graph_topo::GraphTopo topology,
                 std::vector<_N> nodes,
                 std::vector<_E> edges) noexcept
        : _device(std::move(device)),
          _internal(graph_topo::Graph<_N, _E>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    runtime::Stream Graph::lower(Allocator allocator) const {

        runtime::Resources res;

        auto nodeCount = _internal.nodes.size();
        std::vector<size_t> workspace(nodeCount, 0);
        std::vector<runtime::Node> nodes;
        nodes.reserve(nodeCount);
        for (auto i : range0_(nodeCount)) {
            if (auto const &node = _internal.nodes[i]; node.kernel) {
                nodes.emplace_back(node.kernel->lower(res));
            } else {
                nodes.emplace_back(runtime::emptyRoutine);
            }
        }

        auto [stack, nodes_, edges_] = allocator(
            _internal.topology,
            std::move(nodes),
            _internal.edges,
            32);

        for (auto i : range0_(edges_.size())) {
            auto const &edge = _internal.edges[i];
            if (edge.data) {
                edges_[i].blob = edge.data;
            } else if (edges_[i].stackOffset == SIZE_MAX - 1) {
                edges_[i].blob = _device->malloc(_internal.edges[i].size);
            }
        }


        return runtime::Stream(
            std::move(res),
            stack,
            _internal.topology,
            std::move(nodes_),
            std::move(edges_),
            _device);
    }

}// namespace refactor::kernel
