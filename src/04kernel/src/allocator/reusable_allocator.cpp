#include "reusable_allocator.h"
#include "common.h"
#include "mem_manager/mem_offset_calculator.h"

namespace refactor::kernel {

    AllocScheme reusableAllocate(graph_topo::Graph<Node, Edge> const &g, size_t alignBits) {
        // 下划线命名表示什么？
        auto globalOutputs_ = g.topology.globalOutputs();
        std::unordered_set<size_t> globalOutputs(globalOutputs_.begin(), globalOutputs_.end());
        std::vector<Address> addresses(g.edges.size(), {nullptr});
        std::unordered_map<size_t, size_t> edgeToRefCount;
        auto searcher = graph_topo::Searcher(g.topology);
        // 需要传入对齐值，单位统一为字节？
        mem_manager::OffsetCalculator calculator = mem_manager::OffsetCalculator(alignBits);
        for (auto edge : searcher.edges()) {
            edgeToRefCount[edge.index()] = edge.targets().size();
        }
        for (auto [nodeIdx, inputs, outputs] : g.topology) {
            for (auto outputIdx : outputs) {
                if (globalOutputs.find(outputIdx) == globalOutputs.end()) {
                    addresses[outputIdx] = {calculator.alloc(g.edges[outputIdx].size)};
                }
            }
            for (auto inputIdx : inputs) {
                auto edgeIter = edgeToRefCount.find(inputIdx);
                ASSERT(edgeIter != edgeToRefCount.end(),
                       "unknown edge idx in searcher.edges(): " + std::to_string(inputIdx));
                ASSERT(edgeToRefCount[inputIdx] > 0, "double free");
                edgeToRefCount[inputIdx] -= 1;
                if (edgeToRefCount[inputIdx] == 0) {
                    // indicate that this tensor will no longer be used and
                    // perform memory free
                    edgeToRefCount.erase(inputIdx);
                    if (addresses[inputIdx].isOffset()) {
                        calculator.free(addresses[inputIdx].getOffset(), g.edges[inputIdx].size);
                    }
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
        return {calculator.getPeek(), std::move(addresses)};
    }

}// namespace refactor::kernel
