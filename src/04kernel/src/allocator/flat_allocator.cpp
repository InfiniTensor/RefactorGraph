#include "flat_allocator.h"
#include "common/error_handler.h"

namespace refactor::kernel {

    constexpr size_t align(size_t size, int bits) {
        auto mask = (1 << bits) - 1;
        return (size + mask) & ~mask;
    }

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &g) {
        constexpr static auto INVALID = SIZE_MAX;
        size_t size = 0,
               globalInputsCount = g.topology.globalInputsCount();
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<size_t> offsets(g.edges.size(), INVALID);
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto i : outputs) {
                if (globalOutputs.find(i) == globalOutputs.end()) {
                    offsets[i] = size;
                    size += align(g.edges[i].size, 8);
                }
            }
        }
        return {size, std::move(offsets)};
    }

}// namespace refactor::kernel
