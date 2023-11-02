#include "kernel/allocators.h"
#include "mem_manager/mem_offset_calculator.h"

namespace refactor::kernel {

    AllocScheme reusableAllocate(graph_topo::Graph<Node, Edge> const &g, size_t alignBytes) {
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<Address> addresses(g.edges.size(), {nullptr});
        // counts edges reference
        std::vector<size_t> edgeRc;
        for (auto edge : g.topology.connections()) {
            if (edge >= edgeRc.size()) {
                edgeRc.resize(edge + 1, 0);
            }
            ++edgeRc[edge];
        }
        mem_manager::OffsetCalculator calculator(alignBytes);
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto outputIdx : outputs) {
                if (globalOutputs.find(outputIdx) == globalOutputs.end()) {
                    addresses[outputIdx] = {calculator.alloc(g.edges[outputIdx].size)};
                }
            }
            for (auto inputIdx : inputs) {
                ASSERT(edgeRc[inputIdx], "double free");
                if (!--edgeRc[inputIdx]) {
                    // indicate that this tensor will no longer be used and perform memory free
                    if (addresses[inputIdx].isOffset()) {
                        calculator.free(addresses[inputIdx].offset(), g.edges[inputIdx].size);
                    }
                }
            }
        }
        for (auto i : range0_(addresses.size())) {
            if (addresses[i].isBlob()) {
                addresses[i] = {g.edges[i].data};
            }
        }
        return {calculator.peak(), std::move(addresses)};
    }

}// namespace refactor::kernel
