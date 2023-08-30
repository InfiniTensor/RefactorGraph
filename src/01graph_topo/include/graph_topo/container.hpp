#ifndef GRAPH_TOPO_CONTAINER_HPP
#define GRAPH_TOPO_CONTAINER_HPP

#include <cstddef>
#include <vector>

namespace refactor::graph_topo {
    class Searcher;
    class Builder;

    class GraphTopo {
        friend class Searcher;
        friend class Builder;

        class __Implement;
        __Implement *_impl;

    public:
        GraphTopo();
        GraphTopo(GraphTopo const &);
        GraphTopo(GraphTopo &&) noexcept;
        ~GraphTopo();

        GraphTopo &operator=(GraphTopo const &);
        GraphTopo &operator=(GraphTopo &&) noexcept;
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_CONTAINER_HPP
