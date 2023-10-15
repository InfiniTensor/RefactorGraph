#include "graph_topo/searcher.h"
#include "refactor/common.h"
#include <algorithm>
#include <unordered_set>
#include <utility>

namespace refactor::graph_topo {
    constexpr static idx_t EXTERNAL = std::numeric_limits<idx_t>::max();

    Searcher::Searcher(GraphTopo const &graph) noexcept
        : _graph(graph),
          _nodes(graph._nodes.size()),
          _edges(graph._lenIn, {EXTERNAL, {}}),
          _localEdges{} {

        auto passConnections = graph._lenOut;
        for (auto i : range0_(static_cast<idx_t>(_nodes.size()))) {
            auto const &node = graph._nodes[i];
            auto passEdges = static_cast<idx_t>(_edges.size());
            // update _nodes
            _nodes[i] = {passEdges, passConnections, {}, {}};
            // update _edges
            for (auto end = passConnections + node._inputsCount;
                 passConnections != end;
                 ++passConnections) {
                if (auto j = graph._connections[passConnections]; j < _edges.size()) {
                    _edges[j]._targets.insert(i);
                }
            }
            _edges.reserve(passEdges + node._localEdgesCount + node._outputsCount);
            _edges.resize(_edges.size() + node._localEdgesCount, {EXTERNAL, {i}});
            _edges.resize(_edges.size() + node._outputsCount, {i, {}});
            // update _localEdges
            auto local = range(passEdges, passEdges + node._localEdgesCount);
            _localEdges.insert(local.begin(), local.end());
        }
        for (auto edgeIdx : graph.globalOutputs()) {
            _edges[edgeIdx]._targets.insert(EXTERNAL);
        }
    }

    auto Searcher::nodes() const noexcept -> Nodes { return {*this}; }
    auto Searcher::edges() const noexcept -> Edges { return {*this}; }
    auto Searcher::globalInputs() const noexcept -> std::vector<Edge> {
        auto globalInputs = _graph.globalInputs();
        std::vector<Edge> ans;
        ans.reserve(globalInputs.size());
        std::transform(globalInputs.begin(), globalInputs.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }
    auto Searcher::globalOutputs() const noexcept -> std::vector<Edge> {
        auto globalOutputs = _graph.globalOutputs();
        std::vector<Edge> ans;
        ans.reserve(globalOutputs.size());
        std::transform(globalOutputs.begin(), globalOutputs.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }
    auto Searcher::localEdges() const noexcept -> std::vector<Edge> {
        std::vector<Edge> ans;
        ans.reserve(_localEdges.size());
        std::transform(_localEdges.begin(), _localEdges.end(), std::back_inserter(ans),
                       [this](auto edgeIdx) { return Edge(*this, edgeIdx); });
        return ans;
    }

#define COMPARE(NAME) bool Searcher::NAME::operator

    Searcher::Node::Node(Searcher const &internal, idx_t idx) noexcept : _internal(internal), _idx(idx) {}
    Searcher::Edge::Edge(Searcher const &internal, idx_t idx) noexcept : _internal(internal), _idx(idx) {}
    COMPARE(Node) == (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    COMPARE(Node) != (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    COMPARE(Node) < (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    COMPARE(Node) > (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    COMPARE(Node) <= (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    COMPARE(Node) >= (Node const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    COMPARE(Edge) == (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    COMPARE(Edge) != (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    COMPARE(Edge) < (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    COMPARE(Edge) > (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    COMPARE(Edge) <= (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    COMPARE(Edge) >= (Edge const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    idx_t Searcher::Node::index() const noexcept { return _idx; }
    idx_t Searcher::Edge::index() const noexcept { return _idx; }
    Searcher::Node::operator idx_t() const noexcept { return _idx; }
    Searcher::Edge::operator idx_t() const noexcept { return _idx; }

    auto Searcher::Node::inputs() const noexcept -> std::vector<Edge> {
        auto const &nodeIn = _internal._graph._nodes[_idx];
        auto const &nodeEx = _internal._nodes[_idx];
        auto const &connections = _internal._graph._connections.data();
        std::vector<Edge> ans;
        ans.reserve(nodeIn._inputsCount);
        for (auto edgeIdx : slice(connections + nodeEx._passConnections, nodeIn._inputsCount)) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::outputs() const noexcept -> std::vector<Edge> {
        auto const &nodeIn = _internal._graph._nodes[_idx];
        auto const &nodeEx = _internal._nodes[_idx];
        std::vector<Edge> ans;
        ans.reserve(nodeIn._outputsCount);
        auto begin = nodeEx._passEdges + nodeIn._localEdgesCount,
             end = begin + nodeIn._outputsCount;
        for (auto edgeIdx : range(begin, end)) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::predecessors() const noexcept -> std::set<Node> {
        auto &predecessors = _internal._nodes[_idx]._predecessors;
        std::set<Node> ans;
        if (predecessors.empty()) {
            auto const &nodeIn = _internal._graph._nodes[_idx];
            auto const &nodeEx = _internal._nodes[_idx];
            auto const &connections = _internal._graph._connections.data();
            for (auto edgeIdx : slice(connections + nodeEx._passConnections, nodeIn._inputsCount)) {
                auto nodeIdx = _internal._edges[edgeIdx]._source;
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
    auto Searcher::Node::successors() const noexcept -> std::set<Node> {
        auto &successors = _internal._nodes[_idx]._successors;
        std::set<Node> ans;
        if (successors.empty()) {
            auto const &nodeIn = _internal._graph._nodes[_idx];
            auto const &nodeEx = _internal._nodes[_idx];
            auto begin = nodeEx._passEdges + nodeIn._localEdgesCount,
                 end = begin + nodeIn._outputsCount;
            for (auto edgeIdx : range(begin, end)) {
                for (auto nodeIdx : _internal._edges[edgeIdx]._targets) {
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
    auto Searcher::Edge::source() const noexcept -> Node {
        return {_internal, _internal._edges[_idx]._source};
    }
    auto Searcher::Edge::targets() const noexcept -> std::set<Node> {
        auto const targets = _internal._edges[_idx]._targets;
        std::set<Node> ans;
        for (auto nodeIdx : targets) {
            ans.emplace(_internal, nodeIdx);
        }
        return ans;
    }

    Searcher::Nodes::Nodes(Searcher const &internal) noexcept : _internal(internal) {}
    Searcher::Edges::Edges(Searcher const &internal) noexcept : _internal(internal) {}
    auto Searcher::Nodes::begin() const noexcept -> Iterator { return {_internal, 0}; }
    auto Searcher::Edges::begin() const noexcept -> Iterator { return {_internal, 0}; }
    auto Searcher::Nodes::end() const noexcept -> Iterator { return {_internal, size()}; }
    auto Searcher::Edges::end() const noexcept -> Iterator { return {_internal, size()}; }
    auto Searcher::Nodes::size() const noexcept -> size_t { return _internal._nodes.size(); }
    auto Searcher::Edges::size() const noexcept -> size_t { return _internal._edges.size(); }
    auto Searcher::Nodes::operator[](idx_t idx) const noexcept -> Node { return {_internal, idx}; }
    auto Searcher::Edges::operator[](idx_t idx) const noexcept -> Edge { return {_internal, idx}; }
    auto Searcher::Nodes::at(idx_t idx) const -> Node {
        if (auto s = size(); idx >= s) {
            OUT_OF_RANGE("Searcher::Nodes::at", idx, s);
        }
        return {_internal, idx};
    }
    auto Searcher::Edges::at(idx_t idx) const -> Edge {
        if (auto s = size(); idx >= s) {
            OUT_OF_RANGE("Searcher::Edges::at", idx, s);
        }
        return {_internal, idx};
    }

    Searcher::Nodes::Iterator::Iterator(Searcher const &internal, idx_t idx) noexcept : _internal(internal), _idx(idx) {}
    Searcher::Edges::Iterator::Iterator(Searcher const &internal, idx_t idx) noexcept : _internal(internal), _idx(idx) {}
    COMPARE(Nodes::Iterator) == (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    COMPARE(Nodes::Iterator) != (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    COMPARE(Nodes::Iterator) < (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    COMPARE(Nodes::Iterator) > (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    COMPARE(Nodes::Iterator) <= (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    COMPARE(Nodes::Iterator) >= (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    COMPARE(Edges::Iterator) == (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx == rhs._idx; }
    COMPARE(Edges::Iterator) != (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx != rhs._idx; }
    COMPARE(Edges::Iterator) < (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx < rhs._idx; }
    COMPARE(Edges::Iterator) > (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx > rhs._idx; }
    COMPARE(Edges::Iterator) <= (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx <= rhs._idx; }
    COMPARE(Edges::Iterator) >= (Iterator const &rhs) const noexcept { return &_internal == &rhs._internal && _idx >= rhs._idx; }
    auto Searcher::Nodes::Iterator::operator++() noexcept -> Iterator & {
        ++_idx;
        return *this;
    }
    auto Searcher::Edges::Iterator::operator++() noexcept -> Iterator & {
        ++_idx;
        return *this;
    }
    auto Searcher::Nodes::Iterator::operator*() noexcept -> Node { return {_internal, _idx}; }
    auto Searcher::Edges::Iterator::operator*() noexcept -> Edge { return {_internal, _idx}; }

}// namespace refactor::graph_topo
