#ifndef KERNEL_REUSABLE_ALLOCATOR_H
#define KERNEL_REUSABLE_ALLOCATOR_H

#include "kernel/allocator.h"

namespace refactor::kernel {

    AllocScheme reusableAllocate(graph_topo::Graph<Node, Edge> const &graph, size_t alignBits);

}// namespace refactor::kernel

#endif// KERNEL_REUSABLE_ALLOCATOR_H
