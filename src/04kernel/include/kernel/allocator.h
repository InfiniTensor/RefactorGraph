#ifndef KERNEL_ALLOCATOR_H
#define KERNEL_ALLOCATOR_H

#include "graph.h"

namespace refactor::kernel {

    struct AllocScheme {
        size_t size;
        std::vector<size_t> offsets;
    };

    using Allocator = AllocScheme (*)(graph_topo::Graph<Node, Edge> const &);

}// namespace refactor::kernel

#endif// KERNEL_ALLOCATOR_H
