#include "graph_topo/container.h"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace refactor::graph_topo {

    GraphTopo::Iterator::Iterator(
        GraphTopo const &internal,
        idx_t idx,
        idx_t passConnections,
        idx_t passEdges)
        : _internal(internal),
          _idx(idx),
          _passConnections(passConnections),
          _passEdges(passEdges) {}

    auto GraphTopo::Iterator::begin(GraphTopo const &internal) noexcept -> Iterator {
        return Iterator(internal, 0, internal._lenOut, internal._lenIn);
    }

    auto GraphTopo::Iterator::end(GraphTopo const &internal) noexcept -> Iterator {
        return Iterator(internal, static_cast<idx_t>(internal.nodeCount()), -1, -1);
    }

    bool GraphTopo::Iterator::operator==(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool GraphTopo::Iterator::operator!=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool GraphTopo::Iterator::operator<(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool GraphTopo::Iterator::operator>(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool GraphTopo::Iterator::operator<=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool GraphTopo::Iterator::operator>=(Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }

    auto GraphTopo::Iterator::operator++() noexcept -> Iterator & {
        if (_idx < _internal.nodeCount()) {
            auto const &node = _internal._nodes[_idx++];
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
        if (_idx >= _internal.nodeCount()) {
            OUT_OF_RANGE("Iterator out of range", _idx, _internal.nodeCount());
        }
        auto const &node = _internal._nodes[_idx];
        auto inputsBegin = _internal._connections.data() + _passConnections;
        auto inputsEnd = inputsBegin + node._inputsCount;
        auto outputsBegin = _passEdges + node._localEdgesCount;
        auto outputsEnd = outputsBegin + node._outputsCount;
        return NodeRef{
            _idx,
            {inputsBegin, inputsEnd},
            {outputsBegin, outputsEnd}};
    }

    range_t<idx_t> GraphTopo::Iterator::globalInputs() const noexcept {
        return {0, _internal._lenIn};
    }
    slice_t<idx_t> GraphTopo::Iterator::globalOutputs() const noexcept {
        auto begin = _internal._connections.data();
        return {begin, begin + _internal._lenOut};
    }

    GraphTopo::GraphTopo(idx_t lenIn, idx_t lenOut, size_t lenNode) noexcept
        : _lenIn(lenIn),
          _lenOut(lenOut),
          _connections(lenOut),
          _nodes() {
        _nodes.reserve(lenNode);
    }
    auto GraphTopo::begin() const noexcept -> Iterator { return Iterator::begin(*this); }
    auto GraphTopo::end() const noexcept -> Iterator { return Iterator::end(*this); }
    auto GraphTopo::globalInputsCount() const noexcept -> size_t { return _lenIn; }
    auto GraphTopo::globalOutputsCount() const noexcept -> size_t { return _lenOut; }
    auto GraphTopo::nodeCount() const noexcept -> size_t { return static_cast<idx_t>(_nodes.size()); }
    auto GraphTopo::edgeCount() const noexcept -> size_t {
        return std::accumulate(_nodes.begin(), _nodes.end(), _lenIn,
                               [](auto const acc, auto const &n) { return acc + n._localEdgesCount + n._outputsCount; });
    }
    auto GraphTopo::globalInputs() const noexcept -> range_t<idx_t> {
        return range0_(_lenIn);
    }
    auto GraphTopo::globalOutputs() const noexcept -> slice_t<idx_t> {
        return slice(_connections.data(), static_cast<size_t>(_lenOut));
    }

    std::string GraphTopo::toString() const {
        std::stringstream ss;
        auto it = begin(), end_ = end();
        ss << "*. -> ( ";
        for (auto i : it.globalInputs()) {
            ss << i << ' ';
        }
        ss << ')' << std::endl;
        while (it != end_) {
            auto [nodeIdx, inputs, outputs] = *it++;
            ss << nodeIdx << ". ( ";
            for (auto i : inputs) {
                ss << i << ' ';
            }
            ss << ") -> ( ";
            for (auto i : outputs) {
                ss << i << ' ';
            }
            ss << ')' << std::endl;
        }
        ss << "*. <- ( ";
        for (auto i : it.globalOutputs()) {
            ss << i << ' ';
        }
        ss << ')' << std::endl;
        return ss.str();
    }

}// namespace refactor::graph_topo
