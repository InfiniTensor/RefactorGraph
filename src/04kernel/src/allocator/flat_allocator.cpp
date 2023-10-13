﻿#include "flat_allocator.h"
#include "common/error_handler.h"

namespace refactor::kernel {

    constexpr size_t align(size_t size, int bits) {
        auto mask = (1 << bits) - 1;
        return (size + mask) & ~mask;
    }

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &g) {
        size_t size = 0;
        std::vector<size_t> offsets(g.edges.size());
        for (size_t i = 0; i < offsets.size(); ++i) {
            offsets[i] = size;
            size += align(g.edges[i].size, 8);
        }
        return {size, std::move(offsets)};
    }

}// namespace refactor::kernel