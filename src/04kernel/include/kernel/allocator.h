#ifndef KERNEL_ALLOCATOR_H
#define KERNEL_ALLOCATOR_H

#include "graph.h"
#include "runtime/stream.h"

namespace refactor::kernel {
    using runtime::Address;

    struct AllocScheme {
        size_t size;
        std::vector<Address> addresses;
    };

    using Allocator = AllocScheme (*)(graph_topo::Graph<Node, Edge> const &, size_t);

}// namespace refactor::kernel

#endif// KERNEL_ALLOCATOR_H
