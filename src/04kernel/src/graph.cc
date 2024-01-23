#include "kernel/graph.h"

namespace refactor {
    struct DataKey {
        Arc<hardware::Device> dev;
        Arc<kernel::Blob> blob;
        bool operator==(const DataKey &) const = default;// since C++20
    };
}// namespace refactor

template<>
struct std::hash<refactor::DataKey> {
    std::size_t operator()(refactor::DataKey const &s) const noexcept {
        auto hd = std::hash<decltype(s.dev)>()(s.dev),
             hb = std::hash<decltype(s.blob)>()(s.blob);
        return hd ^ (hb << 1);
    }
};

namespace refactor::kernel {

    Graph::Graph(graph_topo::GraphTopo topology,
                 std::vector<_N> nodes,
                 std::vector<_E> edges) noexcept
        : _internal(graph_topo::Graph<_N, _E>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    runtime::Stream Graph::lower(Arc<hardware::Device> device, Allocator allocator) const {
        device->setContext();
        runtime::Resources res;

        std::vector<runtime::Node> nodes;
        nodes.reserve(_internal.nodes.size());
        for (auto i : range0_(_internal.nodes.size())) {
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

        static std::unordered_map<DataKey, Arc<hardware::Device::Blob>> CACHE;

        for (auto i : range0_(edges_.size())) {
            auto const &edge = _internal.edges[i];
            edges_[i].name = edge.name;
            if (edge.data) {
                auto it = CACHE.find({device, edge.data});
                if (it == CACHE.end()) {
                    auto blob = device->malloc(edge.size);
                    blob->copyFromHost(edge.data->get<void>());
                    std::tie(it, std::ignore) = CACHE.emplace(DataKey{device, edge.data}, std::move(blob));
                }
                edges_[i].blob = it->second;
            } else if (edges_[i].stackOffset == SIZE_MAX - 1) {
                edges_[i].blob = device->malloc(edge.size);
            }
        }

        return runtime::Stream(
            std::move(res),
            stack,
            _internal.topology,
            std::move(nodes_),
            std::move(edges_),
            std::move(device));
    }

}// namespace refactor::kernel
