#include "kernel/allocators.h"
#include "mem_manager/functions.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &g, size_t alignBytes) {
        size_t size = 0;
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<Address> addresses(g.edges.size(), {nullptr});
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto i : outputs) {
                if (globalOutputs.find(i) == globalOutputs.end()) {
                    addresses[i] = {size};
                    size += mem_manager::alignBytes(g.edges[i].size, alignBytes);
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
