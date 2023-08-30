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
        GraphTopo _graph;
        std::vector<EdgeIdx> _globalInputs, _globalOutputs;
        std::unordered_set<EdgeIdx> _localEdges;
        std::vector<__Node> _nodes;
        std::vector<__Edge> _edges;

        __Implement() = default;
        __Implement(__Implement const &) = default;
        __Implement(__Implement &&) noexcept = default;

        __Implement(GraphTopo &&graph)
            : _graph(std::forward<GraphTopo>(graph)),
              _globalInputs(_graph._impl->_globalInputsCount),
              _globalOutputs(),
              _localEdges(),
              _nodes(_graph._impl->_nodes.size()),
              _edges() {
            auto nodesCount = _graph._impl->_nodes.size();
            auto globalInputsCount = _graph._impl->_globalInputsCount;

            auto passConnections = 0;

            for (size_t i = 0; i < globalInputsCount; ++i) {
                _globalInputs[i] = i;
                _edges.push_back({EXTERNAL, {}});
            }
            for (size_t nodeIdx = 0; nodeIdx < nodesCount; ++nodeIdx) {
                auto const &node = _graph._impl->_nodes[nodeIdx];
                for (size_t _ = 0; _ < node._localEdgesCount; ++_) {
                    _localEdges.insert(_edges.size());
                    _edges.push_back({EXTERNAL, {}});
                }
                for (size_t _ = 0; _ < node._outputsCount; ++_) {
                    auto edgeIdx = graph._impl->_connections[passConnections++]._edgeIdx;
                    auto &edge = _edges[edgeIdx];

                    _nodes[nodeIdx]._inputs.push_back(edgeIdx);
                    edge._targets.insert(nodeIdx);

                    if (edge._source != EXTERNAL) {
                        _nodes[nodeIdx]._predecessors.insert(edge._source);
                        _nodes[edge._source]._successors.insert(nodeIdx);
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
        }
    };

    Searcher::Searcher()
        : _impl(nullptr) {}
    Searcher::Searcher(GraphTopo &&graph)
        : _impl(new __Implement(std::forward<GraphTopo>(graph))) {}
    Searcher::Searcher(Searcher const &others)
        : _impl(others._impl ? new __Implement(*others._impl) : nullptr) {}
    Searcher::Searcher(Searcher &&others) noexcept
        : _impl(std::exchange(others._impl, nullptr)) {}
    Searcher::~Searcher() {
        delete std::exchange(_impl, nullptr);
    }

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

}// namespace refactor::graph_topo
