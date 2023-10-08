#ifndef GRAPH_TOPO_INTERNAL_H
#define GRAPH_TOPO_INTERNAL_H

#include "graph_topo/container.h"
#include <cstdint>

namespace refactor::graph_topo {

    struct Node {
        size_t
            _localEdgesCount,
            _inputsCount,
            _outputsCount;
    };

    using OutputEdge = size_t;

    class GraphTopo::__Implement {
    public:
        size_t _globalInputsCount;
        std::vector<Node> _nodes;
        std::vector<OutputEdge> _connections;

        __Implement() noexcept = default;
        __Implement(__Implement const &) noexcept = default;
        __Implement(__Implement &&) noexcept = default;
        __Implement(GraphTopo const &others) noexcept : __Implement(*others._impl) {}
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_INTERNAL_H
