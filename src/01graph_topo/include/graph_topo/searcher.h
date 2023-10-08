#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "container.h"
#include <set>

namespace refactor::graph_topo {

    class Searcher {
        class __Implement;
        __Implement *_impl;

    public:
        class Node;
        class Edge;
        class Nodes;
        class Edges;

        Searcher() noexcept;
        explicit Searcher(GraphTopo const &) noexcept;
        Searcher(Searcher const &) noexcept;
        Searcher(Searcher &&) noexcept;
        ~Searcher() noexcept;

        Searcher &operator=(Searcher const &) noexcept;
        Searcher &operator=(Searcher &&) noexcept;

        Nodes nodes() const noexcept;
        Edges edges() const noexcept;
        std::vector<Edge> globalInputs() const noexcept;
        std::vector<Edge> globalOutputs() const noexcept;
        std::vector<Edge> localEdges() const noexcept;
    };

    class Searcher::Node {
        Searcher const &_internal;
        size_t _idx;

    public:
        Node(Searcher const &, size_t) noexcept;
        bool operator==(Node const &) const noexcept;
        bool operator!=(Node const &) const noexcept;
        bool operator<(Node const &) const noexcept;
        bool operator>(Node const &) const noexcept;
        bool operator<=(Node const &) const noexcept;
        bool operator>=(Node const &) const noexcept;

        size_t index() const noexcept;
        std::vector<Edge> inputs() const noexcept;
        std::vector<Edge> outputs() const noexcept;
        std::set<Node> predecessors() const noexcept;
        std::set<Node> successors() const noexcept;
    };
    class Searcher::Edge {
        Searcher const &_internal;
        size_t _idx;

    public:
        Edge(Searcher const &, size_t) noexcept;
        bool operator==(Edge const &) const noexcept;
        bool operator!=(Edge const &) const noexcept;
        bool operator<(Edge const &) const noexcept;
        bool operator>(Edge const &) const noexcept;
        bool operator<=(Edge const &) const noexcept;
        bool operator>=(Edge const &) const noexcept;

        size_t index() const noexcept;
        Node source() const noexcept;
        std::set<Node> targets() const noexcept;
    };

    class Searcher::Nodes {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            size_t _idx;

        public:
            Iterator(Searcher const &, size_t) noexcept;
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
        Node operator[](size_t) const noexcept;
        Node at(size_t) const;
    };
    class Searcher::Edges {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            size_t _idx;

        public:
            Iterator(Searcher const &, size_t) noexcept;
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
        Edge operator[](size_t idx) const noexcept;
        Edge at(size_t) const;
    };
}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_SEARCHER_HPP
