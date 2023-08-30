#ifndef GRAPH_TOPO_CONTAINER_HPP
#define GRAPH_TOPO_CONTAINER_HPP

#include <cstddef>
#include <vector>

namespace refactor::graph_topo {
    class Searcher;

    class GraphTopo {
        friend class Searcher;

        class __Implement;
        __Implement *_impl;

    public:
        GraphTopo();
        GraphTopo(GraphTopo const &);
        GraphTopo(GraphTopo &&) noexcept;
        ~GraphTopo();

        GraphTopo &operator=(GraphTopo const &);
        GraphTopo &operator=(GraphTopo &&) noexcept;

        static GraphTopo __withGlobalInputs(size_t globalInputsCount);
        void __addNode(size_t newLocalEdgesCount, std::vector<size_t> inputs, size_t outputsCount);
        void __setGlobalOutputs(std::vector<size_t> outputs);
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_CONTAINER_HPP
