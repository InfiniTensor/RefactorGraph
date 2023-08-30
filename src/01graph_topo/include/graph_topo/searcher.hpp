#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "container.hpp"

namespace refactor::graph_topo {

    class Searcher {
        class __Implement;
        __Implement *_impl;

    public:
        class Nodes;
        class Node;
        class Edges;
        class Edge;

        Searcher();
        explicit Searcher(GraphTopo &&);
        Searcher(Searcher const &);
        Searcher(Searcher &&) noexcept;
        ~Searcher();

        Searcher &operator=(Searcher const &);
        Searcher &operator=(Searcher &&) noexcept;
    };

    class Searcher::Nodes {};
    class Searcher::Node {};
    class Searcher::Edges {};
    class Searcher::Edge {};

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_SEARCHER_HPP
