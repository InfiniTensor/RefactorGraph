#ifndef KERNEL_ALLOCATOR_H
#define KERNEL_ALLOCATOR_H

#include "graph.h"

namespace refactor::kernel {

    AllocScheme flatAllocate(
        graph_topo::GraphTopo const &,
        std::vector<runtime::Node>,
        std::vector<Edge> const &,
        size_t);

    AllocScheme reusableAllocate(
        graph_topo::GraphTopo const &,
        std::vector<runtime::Node>,
        std::vector<Edge> const &,
        size_t);

}// namespace refactor::kernel

#endif// KERNEL_ALLOCATOR_H
