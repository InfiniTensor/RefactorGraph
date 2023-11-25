#ifndef KERNEL_ALLOCATOR_H
#define KERNEL_ALLOCATOR_H

#include "graph.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &, std::span<size_t const>, size_t);
    AllocScheme reusableAllocate(graph_topo::Graph<Node, Edge> const &, std::span<size_t const>, size_t);

}// namespace refactor::kernel

#endif// KERNEL_ALLOCATOR_H
