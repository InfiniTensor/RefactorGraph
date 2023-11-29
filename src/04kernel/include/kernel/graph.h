#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "kernel.h"
#include <span>

namespace refactor::kernel {

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

        Arc<hardware::Device> _device;
        _G _internal;

    public:
        Graph(decltype(_device),
              graph_topo::GraphTopo,
              std::vector<_N>,
              std::vector<_E>) noexcept;
        runtime::Stream lower(Allocator) const;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
