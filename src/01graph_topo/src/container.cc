﻿#include "common/error_handler.h"
#include "internal.h"
#include <algorithm>
#include <utility>

namespace refactor::graph_topo {

    GraphTopo::GraphTopo()
        : _impl(new __Implement) {}
    GraphTopo::GraphTopo(GraphTopo const &others)
        : _impl(others._impl ? new __Implement(*others._impl) : nullptr) {}
    GraphTopo::GraphTopo(GraphTopo &&others) noexcept
        : _impl(std::exchange(others._impl, nullptr)) {}
    GraphTopo::~GraphTopo() {
        delete std::exchange(_impl, nullptr);
    }

    auto GraphTopo::operator=(GraphTopo const &others) -> GraphTopo & {
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

    auto GraphTopo::Iterator::begin(GraphTopo const *internal) -> Iterator {
        Iterator ans;
        ans._internal = internal;
        ans._idx = 0;
        ans._passConnections = 0;
        ans._passEdges = internal->_impl->_globalInputsCount;
        return ans;
    }

    auto GraphTopo::Iterator::end(GraphTopo const *internal) -> Iterator {
        Iterator ans;
        ans._internal = internal;
        ans._idx = internal->_impl->_nodes.size();
        ans._passConnections = -1;
        ans._passEdges = -1;
        return ans;
    }

    bool GraphTopo::Iterator::operator==(Iterator const &rhs) const { return _internal == rhs._internal && _idx == rhs._idx; }
    bool GraphTopo::Iterator::operator!=(Iterator const &rhs) const { return _internal == rhs._internal && _idx != rhs._idx; }
    bool GraphTopo::Iterator::operator<(Iterator const &rhs) const { return _internal == rhs._internal && _idx < rhs._idx; }
    bool GraphTopo::Iterator::operator>(Iterator const &rhs) const { return _internal == rhs._internal && _idx > rhs._idx; }
    bool GraphTopo::Iterator::operator<=(Iterator const &rhs) const { return _internal == rhs._internal && _idx <= rhs._idx; }
    bool GraphTopo::Iterator::operator>=(Iterator const &rhs) const { return _internal == rhs._internal && _idx >= rhs._idx; }

    auto GraphTopo::Iterator::operator++() -> Iterator & {
        if (_idx < _internal->_impl->_nodes.size()) {
            auto const &node = _internal->_impl->_nodes[_idx++];
            _passConnections += node._inputsCount;
            _passEdges += node._localEdgesCount + node._outputsCount;
        }
        return *this;
    }

    auto GraphTopo::Iterator::operator*() const -> NodeRef {
        if (_idx >= _internal->_impl->_nodes.size()) {
            OUT_OF_RANGE("Iterator out of range", _idx, _internal->_impl->_nodes.size());
        }
        auto const &node = _internal->_impl->_nodes[_idx];
        std::vector<size_t> inputs(node._inputsCount), outputs(node._outputsCount);
        {
            auto const begin = _internal->_impl->_connections.begin() + _passConnections;
            auto const end = begin + node._inputsCount;
            std::transform(begin, end, inputs.begin(), [](auto const &edge) { return edge._edgeIdx; });
        }
        {
            auto const begin = _passEdges + node._localEdgesCount;
            for (size_t i = 0; i < outputs.size(); ++i) { outputs[i] = begin + i; }
        }
        return NodeRef{_idx, std::move(inputs), std::move(outputs)};
    }

    auto GraphTopo::begin() const -> Iterator { return Iterator::begin(this); }
    auto GraphTopo::end() const -> Iterator { return Iterator::end(this); }
    size_t GraphTopo::size() const { return _impl->_nodes.size(); }

    GraphTopo GraphTopo::__withGlobalInputs(size_t globalInputsCount) {
        GraphTopo ans;
        ans._impl->_globalInputsCount = globalInputsCount;
        return ans;
    }
    void GraphTopo::__addNode(size_t newLocalEdgesCount, std::vector<size_t> inputs, size_t outputsCount) {
        _impl->_nodes.push_back({newLocalEdgesCount, inputs.size(), outputsCount});
        for (auto const &edge : inputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }
    void GraphTopo::__setGlobalOutputs(std::vector<size_t> outputs) {
        for (auto const &edge : outputs) {
            _impl->_connections.push_back(OutputEdge{edge});
        }
    }

}// namespace refactor::graph_topo