#include "flat_allocator.h"
#include "refactor/common.h"

namespace refactor::kernel {

    constexpr size_t align(size_t size, int bits) {
        auto mask = (1 << bits) - 1;
        return (size + mask) & ~mask;
    }

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &g) {
        size_t size = 0;
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<Address> addresses(g.edges.size(), {nullptr});
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto i : outputs) {
                if (globalOutputs.find(i) == globalOutputs.end()) {
                    addresses[i] = {size};
                    size += align(g.edges[i].size, 8);
                }
            }
        }
        for (auto i : range0_(addresses.size())) {
            if (addresses[i].isBlob()) {
                auto blob = g.edges[i].data;
                ASSERT(blob, "Blob not exist");
                addresses[i] = {std::move(blob)};
            }
        }
        return {size, std::move(addresses)};
    }

}// namespace refactor::kernel
