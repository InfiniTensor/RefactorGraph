#include "common/error_handler.h"
#include "internal.h"
#include <algorithm>
#include <numeric>
#include <utility>

namespace refactor::graph_topo {

    GraphTopo::GraphTopo() noexcept
        : _impl(new __Implement) {}
    GraphTopo::GraphTopo(GraphTopo const &others) noexcept
        : _impl(others._impl ? new __Implement(*others._impl) : nullptr) {}
    GraphTopo::GraphTopo(GraphTopo &&others) noexcept
        : _impl(std::exchange(others._impl, nullptr)) {}
    GraphTopo::~GraphTopo() noexcept {
        delete std::exchange(_impl, nullptr);
    }

    auto GraphTopo::operator=(GraphTopo const &others) noexcept -> GraphTopo & {
        if (this != &others) {
            delete std::exchange(_impl, others._impl ? new __Implement(*others._impl) : nullptr);
        }
        return *this;
    }
    auto GraphTopo::operator=(GraphTopo &&others) noexcept -> GraphTopo & {
        if (this != &others) {
            delete std::exchange(_impl, std::exchange(others._impl, nullptr));
        }
        return *this;
    }

    GraphTopo::Iterator::Iterator(
        GraphTopo const &internal,
        size_t idx,
        size_t passConnections,
        size_t passEdges)
        : _internal(internal),
          _idx(idx),
          _passConnections(passConnections),
          _passEdges(passEdges) {}

    auto GraphTopo::Iterator::begin(GraphTopo const *internal) noexcept -> Iterator {
        return Iterator(*internal, 0, 0, internal->_impl->_globalInputsCount);
    }

    auto GraphTopo::Iterator::end(GraphTopo const *internal) noexcept -> Iterator {
        return Iterator(*internal, internal->_impl->_nodes.size(), -1, -1);
    }

    bool GraphTopo::Iterator::operator==(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool GraphTopo::Iterator::operator!=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool GraphTopo::Iterator::operator<(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool GraphTopo::Iterator::operator>(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool GraphTopo::Iterator::operator<=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool GraphTopo::Iterator::operator>=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }

    auto GraphTopo::Iterator::operator++() noexcept -> Iterator & {
        if (_idx < _internal._impl->_nodes.size()) {
            auto const &node = _internal._impl->_nodes[_idx++];
            _passConnections += node._inputsCount;
            _passEdges += node._localEdgesCount + node._outputsCount;
        }
        return *this;
    }

    auto GraphTopo::Iterator::operator++(int) noexcept -> Iterator {
        auto ans = *this;
        operator++();
        return ans;
    }

    auto GraphTopo::Iterator::operator*() const noexcept -> NodeRef {
        if (_idx >= _internal._impl->_nodes.size()) {
            OUT_OF_RANGE("Iterator out of range", _idx, _internal._impl->_nodes.size());
        }
        auto const &node = _internal._impl->_nodes[_idx];
        auto inputsBegin = reinterpret_cast<size_t *>(_internal._impl->_connections.data()) + _passConnections;
        auto inputsEnd = inputsBegin + node._inputsCount;
        auto outputsBegin = _passEdges + node._localEdgesCount;
        auto outputsEnd = outputsBegin + node._outputsCount;
        return NodeRef{
            _idx,
            {inputsBegin, inputsEnd},
            {outputsBegin, outputsEnd}};
    }

    common::range_t<size_t> GraphTopo::Iterator::globalInputs() const noexcept {
        return {0, _internal._impl->_globalInputsCount};
    }
    common::slice_t<size_t> GraphTopo::Iterator::globalOutputs() const noexcept {
        ASSERT(_idx == _internal._impl->_nodes.size(), "Iterator not at end");
        auto const &connections = _internal._impl->_connections;
        auto begin = reinterpret_cast<size_t const *>(connections.data());
        return {begin + _passConnections, begin + connections.size()};
    }

    auto GraphTopo::begin() const noexcept -> Iterator { return Iterator::begin(this); }
    auto GraphTopo::end() const noexcept -> Iterator { return Iterator::end(this); }
    auto GraphTopo::size() const noexcept -> size_t { return _impl->_nodes.size(); }
    auto GraphTopo::globalInputsCount() const noexcept -> size_t { return _impl->_globalInputsCount; }
    auto GraphTopo::nodeCount() const noexcept -> size_t { return _impl->_nodes.size(); }
    auto GraphTopo::edgeCount() const noexcept -> size_t {
        return std::accumulate(_impl->_nodes.begin(), _impl->_nodes.end(), _impl->_globalInputsCount,
                               [](auto const acc, auto const &n) { return acc + n._localEdgesCount + n._outputsCount; });
    }
    auto GraphTopo::globalInputs() const noexcept -> common::range_t<size_t> {
        return common::range0_(_impl->_globalInputsCount);
    }
    auto GraphTopo::globalOutputs() const noexcept -> common::slice_t<size_t> {
        auto i = std::accumulate(_impl->_nodes.begin(), _impl->_nodes.end(), 0,
                                 [](auto const acc, auto const &n) { return acc + n._inputsCount; });
        auto const &connections = _impl->_connections;
        return {connections.data() + i, connections.data() + connections.size()};
    }

    GraphTopo GraphTopo::__withGlobalInputs(
        size_t globalInputsCount) noexcept {
        GraphTopo ans;
        ans._impl->_globalInputsCount = globalInputsCount;
        return ans;
    }
    void GraphTopo::__addNode(
        size_t newLocalEdgesCount,
        std::vector<size_t> inputs,
        size_t outputsCount) noexcept {
        _impl->_nodes.push_back({newLocalEdgesCount, inputs.size(), outputsCount});
        for (auto const &edge : inputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }
    void GraphTopo::__setGlobalOutputs(
        std::vector<size_t> outputs) noexcept {
        for (auto const &edge : outputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }

}// namespace refactor::graph_topo
