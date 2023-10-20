#ifndef GRAPH_TOPO_CONTAINER_H
#define GRAPH_TOPO_CONTAINER_H

#include "common.h"
#include <cstddef>
#include <vector>

namespace refactor::graph_topo {

    using idx_t = uint_lv2;

    class GraphTopo {
        template<class NodeKey, class Node, class EdgeKey, class Edge>
        friend class Builder;
        friend class Searcher;
        friend class InplaceModifier;
        template<class Node, class Edge>
        friend class LinkedGraph;

        struct Node {
            idx_t
                _localEdgesCount,
                _inputsCount,
                _outputsCount;
        };
        using OutputEdge = idx_t;

        idx_t _lenIn, _lenOut;
        std::vector<OutputEdge> _connections;
        std::vector<Node> _nodes;

        GraphTopo(idx_t lenIn, idx_t lenOut, size_t lenNode) noexcept;

    public:
        struct NodeRef {
            idx_t idx;
            slice_t<idx_t> inputs;
            range_t<idx_t> outputs;
        };

        class Iterator {
            GraphTopo const &_internal;
            idx_t _idx, _passConnections, _passEdges;
            Iterator(GraphTopo const &, idx_t, idx_t, idx_t);

        public:
            static Iterator begin(GraphTopo const &) noexcept;
            static Iterator end(GraphTopo const &) noexcept;
            bool operator==(Iterator const &) const noexcept;
            bool operator!=(Iterator const &) const noexcept;
            bool operator<(Iterator const &) const noexcept;
            bool operator>(Iterator const &) const noexcept;
            bool operator<=(Iterator const &) const noexcept;
            bool operator>=(Iterator const &) const noexcept;
            Iterator &operator++() noexcept;
            Iterator operator++(int) noexcept;
            NodeRef operator*() const noexcept;
            range_t<idx_t> globalInputs() const noexcept;
            slice_t<idx_t> globalOutputs() const noexcept;
        };

        Iterator begin() const noexcept;
        Iterator end() const noexcept;
        size_t globalInputsCount() const noexcept;
        size_t globalOutputsCount() const noexcept;
        size_t nodeCount() const noexcept;
        size_t edgeCount() const noexcept;
        range_t<idx_t> globalInputs() const noexcept;
        slice_t<idx_t> globalOutputs() const noexcept;

        std::string toString() const;
    };

    template<class Node, class Edge>
    struct Graph {
        GraphTopo topology;
        std::vector<Node> nodes;
        std::vector<Edge> edges;
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_CONTAINER_H
