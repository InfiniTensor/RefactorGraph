#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "container.hpp"
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

        Searcher();
        explicit Searcher(GraphTopo const &);
        Searcher(Searcher const &);
        Searcher(Searcher &&) noexcept;
        ~Searcher();

        Searcher &operator=(Searcher const &);
        Searcher &operator=(Searcher &&) noexcept;

        Nodes nodes() const;
        Edges edges() const;
        std::vector<Edge> globalInputs() const;
        std::vector<Edge> globalOutputs() const;
        std::vector<Edge> localEdges() const;
    };

    class Searcher::Node {
        Searcher const &_internal;
        size_t _idx;

    public:
        Node(Searcher const &, size_t);
        bool operator==(Node const &) const;
        bool operator!=(Node const &) const;
        bool operator<(Node const &) const;
        bool operator>(Node const &) const;
        bool operator<=(Node const &) const;
        bool operator>=(Node const &) const;

        size_t index() const;
        std::vector<Edge> inputs() const;
        std::vector<Edge> outputs() const;
        std::set<Node> predecessors() const;
        std::set<Node> successors() const;
    };
    class Searcher::Edge {
        Searcher const &_internal;
        size_t _idx;

    public:
        Edge(Searcher const &, size_t);
        bool operator==(Edge const &) const;
        bool operator!=(Edge const &) const;
        bool operator<(Edge const &) const;
        bool operator>(Edge const &) const;
        bool operator<=(Edge const &) const;
        bool operator>=(Edge const &) const;

        size_t index() const;
        Node source() const;
        std::set<Node> targets() const;
    };

    class Searcher::Nodes {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            size_t _idx;

        public:
            Iterator(Searcher const &, size_t);
            bool operator==(Iterator const &) const;
            bool operator!=(Iterator const &) const;
            bool operator<(Iterator const &) const;
            bool operator>(Iterator const &) const;
            bool operator<=(Iterator const &) const;
            bool operator>=(Iterator const &) const;
            Iterator &operator++();
            Node operator*();
        };

        Nodes(Searcher const &);
        Iterator begin() const;
        Iterator end() const;
        size_t size() const;
        Node operator[](size_t) const;
    };
    class Searcher::Edges {
        Searcher const &_internal;

    public:
        class Iterator {
            Searcher const &_internal;
            size_t _idx;

        public:
            Iterator(Searcher const &, size_t);
            bool operator==(Iterator const &) const;
            bool operator!=(Iterator const &) const;
            bool operator<(Iterator const &) const;
            bool operator>(Iterator const &) const;
            bool operator<=(Iterator const &) const;
            bool operator>=(Iterator const &) const;
            Iterator &operator++();
            Edge operator*();
        };

        Edges(Searcher const &);
        Iterator begin() const;
        Iterator end() const;
        size_t size() const;
        Edge operator[](size_t idx) const;
    };
}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_SEARCHER_HPP
