#ifndef GRAPH_TOPO_CONTAINER_H
#define GRAPH_TOPO_CONTAINER_H

#include "common.h"

namespace refactor::graph_topo {

    class GraphTopo {
        template<class NodeKey, class Node, class EdgeKey, class Edge>
        friend class Builder;
        friend class Searcher;
        friend class InplaceModifier;
        template<class Node, class Edge>
        friend class LinkedGraph;

        struct Node {
            count_t
                _localEdgesCount,
                _inputsCount,
                _outputsCount;
        };
        using OutputEdge = count_t;

        count_t _lenIn, _lenOut;
        std::vector<OutputEdge> _connections;
        std::vector<Node> _nodes;

        GraphTopo(count_t lenIn, count_t lenOut, size_t lenNode) noexcept;

    public:
        struct NodeRef {
            count_t idx;
            slice_t<count_t> inputs;
            range_t<count_t> outputs;
        };

        class Iterator {
            GraphTopo const &_internal;
            count_t _idx, _passConnections, _passEdges;
            Iterator(GraphTopo const &, count_t, count_t, count_t);

        public:
            static Iterator begin(GraphTopo const &) noexcept;
            static Iterator end(GraphTopo const &) noexcept;
            bool operator==(Iterator const &) const noexcept;
            bool operator!=(Iterator const &) const noexcept;
            bool operator<(Iterator const &) const noexcept;
            bool operator>(Iterator const &) const noexcept;
            bool operator<=(Iterator const &) const noexcept;
            bool operator>=(Iterator const &) const noexcept;
            Iterator &operator++();
            Iterator operator++(int);
            NodeRef operator*() const;
            range_t<count_t> globalInputs() const noexcept;
            slice_t<count_t> globalOutputs() const noexcept;
        };

        Iterator begin() const noexcept;
        Iterator end() const noexcept;
        size_t globalInputsCount() const noexcept;
        size_t globalOutputsCount() const noexcept;
        size_t nodeCount() const noexcept;
        size_t edgeCount() const noexcept;
        range_t<count_t> globalInputs() const noexcept;
        slice_t<count_t> globalOutputs() const noexcept;
        slice_t<count_t> connections() const noexcept;

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
