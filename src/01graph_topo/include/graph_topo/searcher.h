#ifndef GRAPH_TOPO_SEARCHER_H
#define GRAPH_TOPO_SEARCHER_H

#include "container.h"
#include <set>
#include <unordered_set>

namespace refactor::graph_topo {

    class Searcher {
        using EdgeIdx = idx_t;
        using NodeIdx = idx_t;

        struct __Node {
            idx_t _passEdges, _passConnections;
            mutable std::unordered_set<NodeIdx>
                _predecessors,
                _successors;
        };

        struct __Edge {
            NodeIdx _source;
            std::unordered_set<NodeIdx> _targets;
        };

        GraphTopo const &_graph;
        std::unordered_set<EdgeIdx> _localEdges;
        std::vector<__Node> _nodes;
        std::vector<__Edge> _edges;

    public:
        class Node;
        class Edge;
        class Nodes;
        class Edges;

        explicit Searcher(GraphTopo const &) noexcept;

        Nodes nodes() const noexcept;
        Edges edges() const noexcept;
        std::vector<Edge> globalInputs() const noexcept;
        std::vector<Edge> globalOutputs() const noexcept;
        std::vector<Edge> localEdges() const noexcept;
    };

    class Searcher::Node {
        Searcher const &_internal;
        idx_t _idx;

    public:
        Node(Searcher const &, idx_t) noexcept;
        bool operator==(Node const &) const noexcept;
        bool operator!=(Node const &) const noexcept;
        bool operator<(Node const &) const noexcept;
        bool operator>(Node const &) const noexcept;
        bool operator<=(Node const &) const noexcept;
        bool operator>=(Node const &) const noexcept;

        idx_t index() const noexcept;
        std::vector<Edge> inputs() const noexcept;
        std::vector<Edge> outputs() const noexcept;
        std::set<Node> predecessors() const noexcept;
        std::set<Node> successors() const noexcept;
    };
    class Searcher::Edge {
        Searcher const &_internal;
        idx_t _idx;

    public:
        Edge(Searcher const &, idx_t) noexcept;
        bool operator==(Edge const &) const noexcept;
        bool operator!=(Edge const &) const noexcept;
        bool operator<(Edge const &) const noexcept;
        bool operator>(Edge const &) const noexcept;
        bool operator<=(Edge const &) const noexcept;
        bool operator>=(Edge const &) const noexcept;

        idx_t index() const noexcept;
        Node source() const noexcept;
        std::set<Node> targets() const noexcept;
    };

    class Searcher::Nodes {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            idx_t _idx;

        public:
            Iterator(Searcher const &, idx_t) noexcept;
            bool operator==(Iterator const &) const noexcept;
            bool operator!=(Iterator const &) const noexcept;
            bool operator<(Iterator const &) const noexcept;
            bool operator>(Iterator const &) const noexcept;
            bool operator<=(Iterator const &) const noexcept;
            bool operator>=(Iterator const &) const noexcept;
            Iterator &operator++() noexcept;
            Node operator*() noexcept;
        };

        Nodes(Searcher const &) noexcept;
        Iterator begin() const noexcept;
        Iterator end() const noexcept;
        size_t size() const noexcept;
        Node operator[](idx_t) const noexcept;
        Node at(idx_t) const;
    };
    class Searcher::Edges {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            idx_t _idx;

        public:
            Iterator(Searcher const &, idx_t) noexcept;
            bool operator==(Iterator const &) const noexcept;
            bool operator!=(Iterator const &) const noexcept;
            bool operator<(Iterator const &) const noexcept;
            bool operator>(Iterator const &) const noexcept;
            bool operator<=(Iterator const &) const noexcept;
            bool operator>=(Iterator const &) const noexcept;
            Iterator &operator++() noexcept;
            Edge operator*() noexcept;
        };

        Edges(Searcher const &) noexcept;
        Iterator begin() const noexcept;
        Iterator end() const noexcept;
        size_t size() const noexcept;
        Edge operator[](idx_t idx) const noexcept;
        Edge at(idx_t) const;
    };
}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_SEARCHER_H
