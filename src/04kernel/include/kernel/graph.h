#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "kernel.h"
#include <span>

namespace refactor::kernel {
    using runtime::Address;

    struct Node {
        KernelBox kernel;
        std::string name;
    };

    struct Edge {
        Arc<hardware::Device::Blob> data;
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

        _G _internal;

    public:
        Graph(graph_topo::GraphTopo, std::vector<_N>, std::vector<_E>) noexcept;
        runtime::Stream lower(Allocator) const;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
