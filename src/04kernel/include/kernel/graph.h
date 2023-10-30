#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "graph_topo.h"
#include "kernel.h"
#include "mem_manager/foreign_blob.hh"
#include "target.h"

namespace refactor::kernel {
    using mem_manager::SharedForeignBlob;
    using runtime::Address;

    struct Node {
        KernelBox kernel;
        std::string name;
    };

    struct Edge {
        mem_manager::SharedForeignBlob data;
        size_t size;
        std::string name;
    };

    struct AllocScheme {
        size_t size;
        std::vector<Address> addresses;
    };

    using Allocator = AllocScheme (*)(graph_topo::Graph<Node, Edge> const &, size_t);

    struct Graph {
        using _N = Node;
        using _E = Edge;
        using _G = graph_topo::Graph<_N, _E>;

        Target _target;
        _G _internal;

    public:
        Graph(Target, graph_topo::GraphTopo, std::vector<_N>, std::vector<_E>) noexcept;
        runtime::Stream lower(Allocator) const;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
