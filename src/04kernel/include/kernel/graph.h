#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "blob.hh"
#include "kernel.h"

namespace refactor::kernel {

    struct Node {
        KernelBox kernel;
        std::string name;
    };

    struct Edge {
        Arc<Blob> data;
        size_t size;
        std::string name;
    };

    struct AllocScheme {
        size_t stack;
        std::vector<runtime::Node> nodes;
        std::vector<runtime::Edge> edges;
    };

    using Allocator = AllocScheme (*)(
        graph_topo::GraphTopo const &,
        std::vector<runtime::Node>,
        std::vector<Edge> const &,
        size_t);

    struct Graph {
        using _N = Node;
        using _E = Edge;
        using _G = graph_topo::Graph<_N, _E>;

        _G _internal;

    public:
        Graph(graph_topo::GraphTopo,
              std::vector<_N>,
              std::vector<_E>) noexcept;
        runtime::Stream lower(Arc<hardware::Device>, Allocator) const;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
