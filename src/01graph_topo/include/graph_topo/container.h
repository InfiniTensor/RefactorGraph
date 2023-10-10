#ifndef GRAPH_TOPO_CONTAINER_H
#define GRAPH_TOPO_CONTAINER_H

#include "common/range.h"
#include "common/slice.h"
#include <cstddef>
#include <vector>

namespace refactor::graph_topo {
    class Searcher;
    class Modifier;

    class GraphTopo {
        friend class Searcher;
        friend class Modifier;

        class __Implement;
        __Implement *_impl;

    public:
        GraphTopo() noexcept;
        GraphTopo(GraphTopo const &) noexcept;
        GraphTopo(GraphTopo &&) noexcept;
        ~GraphTopo() noexcept;

        GraphTopo &operator=(GraphTopo const &) noexcept;
        GraphTopo &operator=(GraphTopo &&) noexcept;

        struct NodeRef {
            size_t idx;
            common::slice_t<size_t> inputs;
            common::range_t<size_t> outputs;
        };

        class Iterator {
            GraphTopo const &_internal;
            size_t _idx, _passConnections, _passEdges;
            Iterator(GraphTopo const &, size_t, size_t, size_t);

        public:
            static Iterator begin(GraphTopo const *) noexcept;
            static Iterator end(GraphTopo const *) noexcept;
            bool operator==(Iterator const &) const noexcept;
            bool operator!=(Iterator const &) const noexcept;
            bool operator<(Iterator const &) const noexcept;
            bool operator>(Iterator const &) const noexcept;
            bool operator<=(Iterator const &) const noexcept;
            bool operator>=(Iterator const &) const noexcept;
            Iterator &operator++() noexcept;
            Iterator operator++(int) noexcept;
            NodeRef operator*() const noexcept;
            common::range_t<size_t> globalInputs() const noexcept;
            common::slice_t<size_t> globalOutputs() const noexcept;
        };

        Iterator begin() const noexcept;
        Iterator end() const noexcept;
        size_t size() const noexcept;
        size_t globalInputsCount() const noexcept;
        common::range_t<size_t> globalInputs() const noexcept;
        common::slice_t<size_t> globalOutputs() const noexcept;

        static GraphTopo __withGlobalInputs(size_t globalInputsCount) noexcept;
        void __addNode(size_t newLocalEdgesCount, std::vector<size_t> inputs, size_t outputsCount) noexcept;
        void __setGlobalOutputs(std::vector<size_t> outputs) noexcept;
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_CONTAINER_H
