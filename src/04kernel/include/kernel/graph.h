#ifndef KERNEL_GRAPH_H
#define KERNEL_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "kernel.h"
#include "mem_manager/foreign_blob.hh"
#include "target.h"

namespace refactor::kernel {
    using mem_manager::SharedForeignBlob;

    struct Node {
        KernelBox kernel;
        std::string name;
    };

    struct Edge {
        mem_manager::SharedForeignBlob data;
        size_t size;
        std::string name;
    };

    struct Graph {
        using _N = Node;
        using _E = Edge;
        using _G = graph_topo::PolymorphGraph<_N, _E>;

        Target _target;
        _G _internal;

    public:
        Graph(Target, graph_topo::GraphTopo, std::vector<_N>, std::vector<_E>) noexcept;
        runtime::Stream lower() const;
    };

}// namespace refactor::kernel

#endif// KERNEL_GRAPH_H
