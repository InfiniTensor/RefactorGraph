#include "graph_topo/searcher.hpp"
#include "common/range.h"
#include "common/slice.h"
#include "internal.h"
#include <algorithm>
#include <unordered_set>
#include <utility>

namespace refactor::graph_topo {
    using namespace common;

    using EdgeIdx = size_t;
    using NodeIdx = size_t;
    constexpr static NodeIdx EXTERNAL = SIZE_MAX;

    struct __Node {
        size_t _passEdges, _passConnections;
        mutable std::unordered_set<NodeIdx>
            _predecessors,
            _successors;
    };

    struct __Edge {
        NodeIdx _source;
        std::unordered_set<NodeIdx> _targets;
    };

    class Searcher::__Implement {
    public:
        GraphTopo const &_graph;
        range_t<EdgeIdx> _globalInputs;
        slice_t<EdgeIdx> _globalOutputs;
        std::unordered_set<EdgeIdx> _localEdges;
        std::vector<__Node> _nodes;
        std::vector<__Edge> _edges;

        __Implement() = default;
        __Implement(__Implement const &) = default;
        __Implement(__Implement &&) noexcept = default;

        __Implement(GraphTopo const &graph)
            : _graph(graph),
              _nodes(graph._impl->_nodes.size()),
              _edges(graph._impl->_globalInputsCount, {EXTERNAL, {}}),
              _globalInputs{},
              _globalOutputs{},
              _localEdges{} {

            size_t passConnections = 0;
            auto it = graph.begin();
            while (it != graph.end()) {
                auto [nodeIdx, inputs, outputs] = *it++;
                auto localEdgesCount = graph._impl->_nodes[nodeIdx]._localEdgesCount;

                auto edgeBegin = _edges.size();
                _nodes[nodeIdx]._passEdges = edgeBegin;
                _nodes[nodeIdx]._passConnections = passConnections;

                passConnections += inputs.size();
                _edges.resize(edgeBegin + localEdgesCount + outputs.size(), {nodeIdx, {}});
                std::for_each_n(natural_t(edgeBegin), localEdgesCount, [this](auto i) {
                    _localEdges.insert(i);
                    _edges[i]._source = EXTERNAL;
                });

                for (auto edgeIdx : inputs) {
                    _edges[edgeIdx]._targets.insert(nodeIdx);
                }
            }
            _globalInputs = it.globalInputs();
            _globalOutputs = it.globalOutputs();
            for (auto edgeIdx : _globalOutputs) {
                _edges[edgeIdx]._targets.insert(EXTERNAL);
            }
        }
    };

    Searcher::Searcher()
        : _impl(nullptr) {}
    Searcher::Searcher(GraphTopo const &graph)
        : _impl(new __Implement(graph)) {}
    Searcher::Searcher(Searcher const &others)
        : _impl(others._impl ? new __Implement(*others._impl) : nullptr) {}
    Searcher::Searcher(Searcher &&others) noexcept
        : _impl(std::exchange(others._impl, nullptr)) {}
    Searcher::~Searcher() { delete std::exchange(_impl, nullptr); }

    auto Searcher::operator=(Searcher const &others) -> Searcher & {
        if (this != &others) {
            delete std::exchange(_impl, others._impl ? new __Implement(*others._impl) : nullptr);
        }
        return *this;
    }
    auto Searcher::operator=(Searcher &&others) noexcept -> Searcher & {
        if (this != &others) {
            delete std::exchange(_impl, std::exchange(others._impl, nullptr));
        }
        return *this;
    }

    auto Searcher::nodes() const -> Nodes { return {*this}; }
    auto Searcher::edges() const -> Edges { return {*this}; }
    auto Searcher::globalInputs() const -> std::vector<Edge> {
        auto const &globalInputs = _impl->_globalInputs;
        std::vector<Edge> ans;
        ans.reserve(globalInputs.size());
        std::transform(globalInputs.begin(), globalInputs.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }
    auto Searcher::globalOutputs() const -> std::vector<Edge> {
        auto const &globalOutputs = _impl->_globalOutputs;
        std::vector<Edge> ans;
        ans.reserve(globalOutputs.size());
        std::transform(globalOutputs.begin(), globalOutputs.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }
    auto Searcher::localEdges() const -> std::vector<Edge> {
        auto const &localEdges = _impl->_localEdges;
        std::vector<Edge> ans;
        ans.reserve(localEdges.size());
        std::transform(localEdges.begin(), localEdges.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }

    Searcher::Node::Node(Searcher const &internal, size_t idx) : _internal(internal), _idx(idx) {}
    Searcher::Edge::Edge(Searcher const &internal, size_t idx) : _internal(internal), _idx(idx) {}
    bool Searcher::Node::operator==(Node const &rhs) const { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool Searcher::Node::operator!=(Node const &rhs) const { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool Searcher::Node::operator<(Node const &rhs) const { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool Searcher::Node::operator>(Node const &rhs) const { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool Searcher::Node::operator<=(Node const &rhs) const { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Node::operator>=(Node const &rhs) const { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    bool Searcher::Edge::operator==(Edge const &rhs) const { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool Searcher::Edge::operator!=(Edge const &rhs) const { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool Searcher::Edge::operator<(Edge const &rhs) const { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool Searcher::Edge::operator>(Edge const &rhs) const { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool Searcher::Edge::operator<=(Edge const &rhs) const { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Edge::operator>=(Edge const &rhs) const { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    size_t Searcher::Node::index() const { return _idx; }
    size_t Searcher::Edge::index() const { return _idx; }

    auto Searcher::Node::inputs() const -> std::vector<Edge> {
        auto const &nodeIn = _internal._impl->_graph._impl->_nodes[_idx];
        auto const &nodeEx = _internal._impl->_nodes[_idx];
        auto const &connections = _internal._impl->_graph._impl->_connections.data();
        std::vector<Edge> ans;
        ans.reserve(nodeIn._inputsCount);
        for (auto edgeIdx : slice(connections + nodeEx._passConnections, nodeIn._inputsCount)) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::outputs() const -> std::vector<Edge> {
        auto const &nodeIn = _internal._impl->_graph._impl->_nodes[_idx];
        auto const &nodeEx = _internal._impl->_nodes[_idx];
        std::vector<Edge> ans;
        ans.reserve(nodeIn._outputsCount);
        auto begin = nodeEx._passEdges + nodeIn._localEdgesCount,
             end = begin + nodeIn._outputsCount;
        for (auto edgeIdx : range(begin, end)) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::predecessors() const -> std::set<Node> {
        auto const &impl = *_internal._impl;
        auto &predecessors = impl._nodes[_idx]._predecessors;
        std::set<Node> ans;
        if (predecessors.empty()) {
            auto const &nodeIn = impl._graph._impl->_nodes[_idx];
            auto const &nodeEx = impl._nodes[_idx];
            auto const &connections = impl._graph._impl->_connections.data();
            for (auto edgeIdx : slice(connections + nodeEx._passConnections, nodeIn._inputsCount)) {
                auto nodeIdx = impl._edges[edgeIdx]._source;
                if (nodeIdx != EXTERNAL) {
                    if (auto [it, ok] = predecessors.insert(nodeIdx); ok) {
                        ans.emplace(_internal, nodeIdx);
                    }
                }
            }
        } else {
            for (auto nodeIdx : predecessors) {
                ans.emplace(_internal, nodeIdx);
            }
        }
        return ans;
    }
    auto Searcher::Node::successors() const -> std::set<Node> {
        auto const &impl = *_internal._impl;
        auto &successors = impl._nodes[_idx]._successors;
        std::set<Node> ans;
        if (successors.empty()) {
            auto const &nodeIn = impl._graph._impl->_nodes[_idx];
            auto const &nodeEx = impl._nodes[_idx];
            auto begin = nodeEx._passEdges + nodeIn._localEdgesCount,
                 end = begin + nodeIn._outputsCount;
            for (auto edgeIdx : range(begin, end)) {
                for (auto nodeIdx : impl._edges[edgeIdx]._targets) {
                    if (nodeIdx != EXTERNAL) {
                        if (auto [it, ok] = successors.emplace(nodeIdx); ok) {
                            ans.emplace(_internal, nodeIdx);
                        }
                    }
                }
            }
        } else {
            for (auto nodeIdx : successors) {
                ans.emplace(_internal, nodeIdx);
            }
        }
        return ans;
    }
    auto Searcher::Edge::source() const -> Node {
        return {_internal, _internal._impl->_edges[_idx]._source};
    }
    auto Searcher::Edge::targets() const -> std::set<Node> {
        auto const targets = _internal._impl->_edges[_idx]._targets;
        std::set<Node> ans;
        for (auto nodeIdx : targets) {
            ans.emplace(_internal, nodeIdx);
        }
        return ans;
    }

    Searcher::Nodes::Nodes(Searcher const &internal) : _internal(internal) {}
    Searcher::Edges::Edges(Searcher const &internal) : _internal(internal) {}
    auto Searcher::Nodes::begin() const -> Iterator { return {_internal, 0}; }
    auto Searcher::Edges::begin() const -> Iterator { return {_internal, 0}; }
    auto Searcher::Nodes::end() const -> Iterator { return {_internal, size()}; }
    auto Searcher::Edges::end() const -> Iterator { return {_internal, size()}; }
    auto Searcher::Nodes::size() const -> size_t { return _internal._impl->_nodes.size(); }
    auto Searcher::Edges::size() const -> size_t { return _internal._impl->_edges.size(); }
    auto Searcher::Nodes::operator[](size_t idx) const -> Node { return {_internal, idx}; }
    auto Searcher::Edges::operator[](size_t idx) const -> Edge { return {_internal, idx}; }

    Searcher::Nodes::Iterator::Iterator(Searcher const &internal, size_t idx) : _internal(internal), _idx(idx) {}
    Searcher::Edges::Iterator::Iterator(Searcher const &internal, size_t idx) : _internal(internal), _idx(idx) {}
    bool Searcher::Nodes::Iterator::operator==(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool Searcher::Nodes::Iterator::operator!=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool Searcher::Nodes::Iterator::operator<(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool Searcher::Nodes::Iterator::operator>(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool Searcher::Nodes::Iterator::operator<=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Nodes::Iterator::operator>=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    bool Searcher::Edges::Iterator::operator==(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx == rhs._idx; }
    bool Searcher::Edges::Iterator::operator!=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx != rhs._idx; }
    bool Searcher::Edges::Iterator::operator<(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx < rhs._idx; }
    bool Searcher::Edges::Iterator::operator>(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx > rhs._idx; }
    bool Searcher::Edges::Iterator::operator<=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Edges::Iterator::operator>=(Iterator const &rhs) const { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    auto Searcher::Nodes::Iterator::operator++() -> Iterator & {
        ++_idx;
        return *this;
    }
    auto Searcher::Edges::Iterator::operator++() -> Iterator & {
        ++_idx;
        return *this;
    }
    auto Searcher::Nodes::Iterator::operator*() -> Node { return {_internal, _idx}; }
    auto Searcher::Edges::Iterator::operator*() -> Edge { return {_internal, _idx}; }

}// namespace refactor::graph_topo
