#include "hardware/mem_offset_calculator.h"
#include "kernel/allocators.h"

namespace refactor::kernel {

    AllocScheme reusableAllocate(
        graph_topo::GraphTopo const &topology,
        std::vector<runtime::Node> nodes,
        std::vector<Edge> const &edges,
        size_t alignBytes) {
        // counts edges reference
        std::vector<size_t> edgeRc;
        for (auto edge : topology.connections()) {
            if (edge >= edgeRc.size()) {
                edgeRc.resize(edge + 1, 0);
            }
            ++edgeRc[edge];
        }
        // initialize answer
        hardware::OffsetCalculator calculator(alignBytes, true);
        auto globalOutputs_ = topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<runtime::Edge> edges_(edges.size(), {nullptr, SIZE_MAX});
        for (auto [nodeIdx, inputs, outputs] : topology) {
            for (auto outputIdx : outputs) {
                if (globalOutputs.contains(outputIdx)) {
                    edges_[outputIdx].stackOffset--;
                } else {
                    edges_[outputIdx].stackOffset = calculator.alloc(edges[outputIdx].size);
                }
            }
            if (auto &node = nodes[nodeIdx]; node.workspaceOffset) {
                auto wsSize = node.workspaceOffset;
                calculator.free(node.workspaceOffset = calculator.alloc(wsSize), wsSize);
            }
            for (auto outputIdx : outputs) {
                if (!edgeRc[outputIdx]) {
                    calculator.free(edges_[outputIdx].stackOffset, edges[outputIdx].size);
                }
            }
            for (auto inputIdx : inputs) {
                ASSERT(edgeRc[inputIdx], "double free");
                if (!--edgeRc[inputIdx]) {
                    // indicate that this tensor will no longer be used and perform memory free
                    if (edges_[inputIdx].stackOffset != SIZE_MAX) {
                        calculator.free(edges_[inputIdx].stackOffset, edges[inputIdx].size);
                    }
                }
            }
        }
        return {
            calculator.peak(),
            std::move(nodes),
            std::move(edges_),
        };
    }

}// namespace refactor::kernel
