#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "graph_topo.h"
#include "kernel.h"
#include "hardware/foreign_blob.hh"
#include "target.h"
#include <span>

namespace refactor::kernel {
    using hardware::SharedForeignBlob;
    using runtime::Address;

    struct Node {
        KernelBox kernel;
        std::string name;
    };

    struct Edge {
        hardware::SharedForeignBlob data;
        size_t size;
        std::string name;
    };

    struct AllocScheme {
        size_t size;
        std::vector<Address> addresses;
        std::vector<size_t> workspaceOffsets;
    };

    using Allocator = AllocScheme (*)(
        graph_topo::Graph<Node, Edge> const &,
        std::span<size_t const>,
        size_t);

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
