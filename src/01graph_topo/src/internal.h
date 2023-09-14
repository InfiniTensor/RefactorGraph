#ifndef GRAPH_TOPO_INTERNAL_H
#define GRAPH_TOPO_INTERNAL_H

#include "graph_topo/container.hpp"
#include <cstdint>

namespace refactor::graph_topo {

    struct Node {
        size_t
            _localEdgesCount,
            _inputsCount,
            _outputsCount;
    };

    struct OutputEdge {
        size_t _edgeIdx;
    };

    class GraphTopo::__Implement {
    public:
        size_t _globalInputsCount;
        std::vector<Node> _nodes;
        std::vector<OutputEdge> _connections;

        __Implement() = default;
        __Implement(GraphTopo const &others)
            : _globalInputsCount(_globalInputsCount),
              _nodes(others._impl->_nodes),
              _connections(others._impl->_connections) {}
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_INTERNAL_H
