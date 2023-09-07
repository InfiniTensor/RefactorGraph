#include "graph_topo/searcher.hpp"
#include "internal.h"
#include <unordered_set>
#include <utility>

namespace refactor::graph_topo {
    using EdgeIdx = size_t;
    using NodeIdx = size_t;
    constexpr static NodeIdx EXTERNAL = SIZE_MAX;

    struct __Node {
        std::vector<EdgeIdx>
            _inputs,
            _outputs;
        std::unordered_set<NodeIdx>
            _predecessors,
            _successors;
    };

    struct __Edge {
        NodeIdx _source;
        std::unordered_set<NodeIdx> _targets;
    };

    class Searcher::__Implement {
    public:
        std::vector<EdgeIdx> _globalInputs, _globalOutputs;
        std::unordered_set<EdgeIdx> _localEdges;
        std::vector<__Node> _nodes;
        std::vector<__Edge> _edges;

        __Implement() = default;
        __Implement(__Implement const &) = default;
        __Implement(__Implement &&) noexcept = default;

        __Implement(GraphTopo const &graph)
            : _globalInputs(graph._impl->_globalInputsCount),
              _globalOutputs(),
              _localEdges(),
              _nodes(graph._impl->_nodes.size()),
              _edges() {

            auto nodesCount = _nodes.size();
            auto globalInputsCount = _globalInputs.size();

            auto passConnections = 0;

            for (size_t i = 0; i < globalInputsCount; ++i) {
                _globalInputs[i] = i;
                _edges.push_back({EXTERNAL, {}});
            }
            for (size_t nodeIdx = 0; nodeIdx < nodesCount; ++nodeIdx) {
                auto const &node = graph._impl->_nodes[nodeIdx];
                for (size_t _ = 0; _ < node._localEdgesCount; ++_) {
                    _localEdges.insert(_edges.size());
                    _edges.push_back({EXTERNAL, {}});
                }
                for (size_t _ = 0; _ < node._outputsCount; ++_) {
                    _nodes[nodeIdx]._outputs.push_back(_edges.size());
                    _edges.push_back({nodeIdx, {}});
                }
                for (size_t _ = 0; _ < node._inputsCount; ++_) {
                    auto edgeIdx = graph._impl->_connections[passConnections++]._edgeIdx;
                    auto &edge = _edges[edgeIdx];

                    _nodes[nodeIdx]._inputs.push_back(edgeIdx);
                    edge._targets.insert(nodeIdx);

                    if (edge._source != EXTERNAL) {
                        _nodes[nodeIdx]._predecessors.insert(edge._source);
                        _nodes[edge._source]._successors.insert(nodeIdx);
                    }
                }
            }
            auto const &connections = graph._impl->_connections;
            for (auto output = connections.begin() + passConnections; output != connections.end(); ++output) {
                auto edgeIdx = output->_edgeIdx;
                auto &edge = _edges[edgeIdx];

                _globalOutputs.push_back(edgeIdx);
                edge._targets.insert(EXTERNAL);

                if (edge._source != EXTERNAL) {
                    _nodes[edge._source]._successors.insert(EXTERNAL);
                }
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

    auto Searcher::nodes() const -> Nodes { return {this}; }
    auto Searcher::edges() const -> Edges { return {this}; }
    auto Searcher::globalInputs() const -> std::vector<Edge> {
        auto const &globalInputs = _impl->_globalInputs;
        std::vector<Edge> ans;
        ans.reserve(globalInputs.size());
        for (auto edgeIdx : globalInputs) {
            ans.emplace_back(this, edgeIdx);
        }
        return ans;
    }
    auto Searcher::globalOutputs() const -> std::vector<Edge> {
        auto const &globalOutputs = _impl->_globalOutputs;
        std::vector<Edge> ans;
        ans.reserve(globalOutputs.size());
        for (auto edgeIdx : globalOutputs) {
            ans.emplace_back(this, edgeIdx);
        }
        return ans;
    }
    auto Searcher::localEdges() const -> std::vector<Edge> {
        auto const &localEdges = _impl->_localEdges;
        std::vector<Edge> ans;
        ans.reserve(localEdges.size());
        for (auto edgeIdx : localEdges) {
            ans.emplace_back(this, edgeIdx);
        }
        return ans;
    }

    Searcher::Node::Node(Searcher const *internal, size_t idx) : _internal(internal), _idx(idx) {}
    Searcher::Edge::Edge(Searcher const *internal, size_t idx) : _internal(internal), _idx(idx) {}
    bool Searcher::Node::operator==(Node const &rhs) const { return _internal == rhs._internal && _idx == rhs._idx; }
    bool Searcher::Node::operator!=(Node const &rhs) const { return _internal == rhs._internal && _idx != rhs._idx; }
    bool Searcher::Node::operator<(Node const &rhs) const { return _internal == rhs._internal && _idx < rhs._idx; }
    bool Searcher::Node::operator>(Node const &rhs) const { return _internal == rhs._internal && _idx > rhs._idx; }
    bool Searcher::Node::operator<=(Node const &rhs) const { return _internal == rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Node::operator>=(Node const &rhs) const { return _internal == rhs._internal && _idx >= rhs._idx; }
    bool Searcher::Edge::operator==(Edge const &rhs) const { return _internal == rhs._internal && _idx == rhs._idx; }
    bool Searcher::Edge::operator!=(Edge const &rhs) const { return _internal == rhs._internal && _idx != rhs._idx; }
    bool Searcher::Edge::operator<(Edge const &rhs) const { return _internal == rhs._internal && _idx < rhs._idx; }
    bool Searcher::Edge::operator>(Edge const &rhs) const { return _internal == rhs._internal && _idx > rhs._idx; }
    bool Searcher::Edge::operator<=(Edge const &rhs) const { return _internal == rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Edge::operator>=(Edge const &rhs) const { return _internal == rhs._internal && _idx >= rhs._idx; }
    size_t Searcher::Node::index() const { return _idx; }
    size_t Searcher::Edge::index() const { return _idx; }

    auto Searcher::Node::inputs() const -> std::vector<Edge> {
        auto const &inputs = _internal->_impl->_nodes[_idx]._inputs;
        std::vector<Edge> ans;
        ans.reserve(inputs.size());
        for (auto edgeIdx : inputs) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::outputs() const -> std::vector<Edge> {
        auto const &outputs = _internal->_impl->_nodes[_idx]._outputs;
        std::vector<Edge> ans;
        ans.reserve(outputs.size());
        for (auto edgeIdx : outputs) {
            ans.emplace_back(_internal, edgeIdx);
        }
        return ans;
    }
    auto Searcher::Node::predecessors() const -> std::set<Node> {
        auto const predecessors = _internal->_impl->_nodes[_idx]._predecessors;
        std::set<Node> ans;
        for (auto nodeIdx : predecessors) {
            ans.emplace(_internal, nodeIdx);
        }
        return ans;
    }
    auto Searcher::Node::successors() const -> std::set<Node> {
        auto const successors = _internal->_impl->_nodes[_idx]._successors;
        std::set<Node> ans;
        for (auto nodeIdx : successors) {
            ans.emplace(_internal, nodeIdx);
        }
        return ans;
    }
    auto Searcher::Edge::source() const -> Node {
        return {_internal, _internal->_impl->_edges[_idx]._source};
    }
    auto Searcher::Edge::targets() const -> std::set<Node> {
        auto const targets = _internal->_impl->_edges[_idx]._targets;
        std::set<Node> ans;
        for (auto nodeIdx : targets) {
            ans.emplace(_internal, nodeIdx);
        }
        return ans;
    }

    Searcher::Nodes::Nodes(Searcher const *internal) : _internal(internal) {}
    Searcher::Edges::Edges(Searcher const *internal) : _internal(internal) {}
    auto Searcher::Nodes::begin() const -> Iterator { return {_internal, 0}; }
    auto Searcher::Edges::begin() const -> Iterator { return {_internal, 0}; }
    auto Searcher::Nodes::end() const -> Iterator { return {_internal, size()}; }
    auto Searcher::Edges::end() const -> Iterator { return {_internal, size()}; }
    auto Searcher::Nodes::size() const -> size_t { return _internal->_impl->_nodes.size(); }
    auto Searcher::Edges::size() const -> size_t { return _internal->_impl->_edges.size(); }
    auto Searcher::Nodes::operator[](size_t idx) const -> Node { return {_internal, idx}; }
    auto Searcher::Edges::operator[](size_t idx) const -> Edge { return {_internal, idx}; }

    Searcher::Nodes::Iterator::Iterator(Searcher const *internal, size_t idx) : _internal(internal), _idx(idx) {}
    Searcher::Edges::Iterator::Iterator(Searcher const *internal, size_t idx) : _internal(internal), _idx(idx) {}
    bool Searcher::Nodes::Iterator::operator==(Iterator const &rhs) const { return _internal == rhs._internal && _idx == rhs._idx; }
    bool Searcher::Nodes::Iterator::operator!=(Iterator const &rhs) const { return _internal == rhs._internal && _idx != rhs._idx; }
    bool Searcher::Nodes::Iterator::operator<(Iterator const &rhs) const { return _internal == rhs._internal && _idx < rhs._idx; }
    bool Searcher::Nodes::Iterator::operator>(Iterator const &rhs) const { return _internal == rhs._internal && _idx > rhs._idx; }
    bool Searcher::Nodes::Iterator::operator<=(Iterator const &rhs) const { return _internal == rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Nodes::Iterator::operator>=(Iterator const &rhs) const { return _internal == rhs._internal && _idx >= rhs._idx; }
    bool Searcher::Edges::Iterator::operator==(Iterator const &rhs) const { return _internal == rhs._internal && _idx == rhs._idx; }
    bool Searcher::Edges::Iterator::operator!=(Iterator const &rhs) const { return _internal == rhs._internal && _idx != rhs._idx; }
    bool Searcher::Edges::Iterator::operator<(Iterator const &rhs) const { return _internal == rhs._internal && _idx < rhs._idx; }
    bool Searcher::Edges::Iterator::operator>(Iterator const &rhs) const { return _internal == rhs._internal && _idx > rhs._idx; }
    bool Searcher::Edges::Iterator::operator<=(Iterator const &rhs) const { return _internal == rhs._internal && _idx <= rhs._idx; }
    bool Searcher::Edges::Iterator::operator>=(Iterator const &rhs) const { return _internal == rhs._internal && _idx >= rhs._idx; }
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
