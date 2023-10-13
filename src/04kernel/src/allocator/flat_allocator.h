#ifndef KERNEL_FLAT_ALLOCATOR_H
#define KERNEL_FLAT_ALLOCATOR_H

#include "kernel/allocator.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(graph_topo::Graph<Node, Edge> const &graph);

}// namespace refactor::kernel

#endif// KERNEL_FLAT_ALLOCATOR_H
