#include "hardware/functions.h"
#include "kernel/allocators.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(
        graph_topo::GraphTopo const &topology,
        std::vector<runtime::Node> nodes,
        std::vector<Edge> const &edges,
        size_t alignBytes) {
        // check usage of edges
        std::vector<bool> used;
        for (auto edge : topology.connections()) {
            if (edge >= used.size()) {
                used.resize(edge + 1, false);
            }
            used[edge] = true;
        }
        // initialize answer
        size_t size = 0;
        auto globalOutputs_ = topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<runtime::Edge> edges_(edges.size(), {nullptr, SIZE_MAX});
        for (auto [nodeIdx, inputs, outputs] : topology) {
            for (auto i : outputs) {
                if (!used[i]) { continue; }
                if (globalOutputs.contains(i)) {
                    edges_[i].stackOffset--;
                } else {
                    edges_[i].stackOffset = size;
                    size += hardware::alignBytes(edges[i].size, alignBytes);
                }
            }
            if (auto &node = nodes[nodeIdx]; node.workspaceOffset) {
                size += hardware::alignBytes(std::exchange(node.workspaceOffset, size), alignBytes);
            }
        }
        return {
            size,
            std::move(nodes),
            std::move(edges_),
        };
    }

}// namespace refactor::kernel
