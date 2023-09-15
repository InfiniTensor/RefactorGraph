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

        struct NodeRef {
            size_t idx;
            std ::vector<size_t> inputs, outputs;
        };

        class Iterator {
            GraphTopo const *_internal;
            size_t _idx, _passConnections, _passEdges;

        public:
            static Iterator begin(GraphTopo const *);
            static Iterator end(GraphTopo const *);
            bool operator==(Iterator const &) const;
            bool operator!=(Iterator const &) const;
            bool operator<(Iterator const &) const;
            bool operator>(Iterator const &) const;
            bool operator<=(Iterator const &) const;
            bool operator>=(Iterator const &) const;
            Iterator &operator++();
            Iterator operator++(int);
            NodeRef operator*() const;
            std::vector<size_t> globalInputs() const;
            std::vector<size_t> globalOutputs() const;
        };

        Iterator begin() const;
        Iterator end() const;
        size_t size() const;
        size_t globalInputsCount() const;

        static GraphTopo __withGlobalInputs(size_t globalInputsCount);
        void __addNode(size_t newLocalEdgesCount, std::vector<size_t> inputs, size_t outputsCount);
        void __setGlobalOutputs(std::vector<size_t> outputs);
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_CONTAINER_HPP
