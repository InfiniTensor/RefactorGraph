#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "kernel.h"
#include "mem_manager/foreign_blob.hh"

namespace refactor::kernel {
    using mem_manager::SharedForeignBlob;

    struct Node {
        std::unique_ptr<Kernel> kernel;
        std::string name;
    };

    struct Edge {
        mem_manager::SharedForeignBlob data;
        std::string name;
    };

    struct Graph {
        using _N = Node;
        using _E = Edge;
        using _G = graph_topo::Graph<_N, _E>;

        _G _internal;

    public:
        explicit Graph(_G) noexcept;
        Graph(graph_topo::GraphTopo, std::vector<_N>, std::vector<_E>) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
