#include "kernel/allocators.h"
#include "hardware/functions.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &g,
                             std::span<size_t const> workspace,
                             size_t alignBytes) {
        // check usage of edges
        std::vector<bool> used;
        for (auto edge : g.topology.connections()) {
            if (edge >= used.size()) {
                used.resize(edge + 1, false);
            }
            used[edge] = true;
        }
        // initialize answer
        size_t size = 0;
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<Address> addresses(g.edges.size(), {nullptr});
        std::vector<size_t> workspaceOffsets(workspace.size(), SIZE_MAX);
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto i : outputs) {
                if (used[i] && !globalOutputs.contains(i)) {
                    addresses[i] = {size};
                    size += hardware::alignBytes(g.edges[i].size, alignBytes);
                }
            }
            if (auto wsSize = workspace[nodeIdx]; wsSize) {
                workspaceOffsets[nodeIdx] = size;
                size += hardware::alignBytes(wsSize, alignBytes);
            }
        }
        for (auto i : range0_(addresses.size())) {
            if (addresses[i].isBlob()) {
                addresses[i] = {g.edges[i].data};
            }
        }
        return {
            size,
            std::move(addresses),
            std::move(workspaceOffsets),
        };
    }

}// namespace refactor::kernel
