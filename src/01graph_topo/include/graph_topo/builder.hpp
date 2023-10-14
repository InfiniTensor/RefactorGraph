#ifndef GRAPH_TOPO_BUILDER_HPP
#define GRAPH_TOPO_BUILDER_HPP

#include "container.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace refactor::graph_topo {

    template<class EdgeKey>
    struct BuilderNode {
        std::vector<EdgeKey> inputs, outputs;
    };

    template<class NodeKey, class Node, class EdgeKey, class Edge>
    struct Builder {
        std::unordered_map<NodeKey, BuilderNode<EdgeKey>> topology;
        std::vector<EdgeKey> globalInputs, globalOutputs;
        std::unordered_map<NodeKey, Node> nodes;
        std::unordered_map<EdgeKey, Edge> edges;

        Graph<Node, Edge> build() noexcept {
            auto topology = GraphTopo(
                static_cast<idx_t>(globalInputs.size()),
                static_cast<idx_t>(globalOutputs.size()),
                this->topology.size());
            std::vector<Node> nodes;
            std::vector<Edge> edges;

            std::unordered_map<EdgeKey, idx_t> keyToIdx;
            auto mapEdge = [&keyToIdx, &edges, this](EdgeKey const &edge) {
                keyToIdx[edge] = static_cast<idx_t>(edges.size());
                if (auto it = this->edges.find(edge); it != this->edges.end()) {
                    edges.emplace_back(std::move(it->second));
                } else {
                    edges.emplace_back();
                }
            };
            std::unordered_set<EdgeKey> notLocal;
            for (auto const &edge : globalInputs) {
                mapEdge(edge);
                notLocal.insert(edge);
            }
            auto connectionCount = globalOutputs.size();
            for (auto const &[_, value] : this->topology) {
                connectionCount += value.inputs.size();
                notLocal.insert(value.outputs.begin(), value.outputs.end());
            }
            topology._connections.reserve(connectionCount);

            std::unordered_set<NodeKey> mappedNodes;
            while (mappedNodes.size() < this->topology.size()) {
                for (auto const &[kn, n__] : this->topology) {
                    // node mapped ?
                    if (mappedNodes.find(kn) != mappedNodes.end()) {
                        continue;
                    }
                    // all inputs mapped ?
                    auto const &[inputs, outputs] = n__;
                    auto ok = true;
                    std::unordered_set<EdgeKey> newLocal;
                    for (auto const &edge : inputs) {
                        if (keyToIdx.find(edge) != keyToIdx.end()) {
                            continue;
                        }
                        if (notLocal.find(edge) != notLocal.end()) {
                            ok = false;
                            break;
                        }
                        newLocal.insert(edge);
                    }
                    if (!ok) {
                        continue;
                    }
                    // map node
                    mappedNodes.insert(kn);
                    nodes.emplace_back(std::move(this->nodes.at(kn)));
                    topology._nodes.push_back({
                        static_cast<idx_t>(newLocal.size()),
                        static_cast<idx_t>(inputs.size()),
                        static_cast<idx_t>(outputs.size()),
                    });
                    // map edges
                    for (auto const &edge : newLocal) { mapEdge(edge); }
                    for (auto const &edge : outputs) { mapEdge(edge); }
                    // map connections
                    for (auto const &input : inputs) {
                        topology._connections.push_back(keyToIdx.at(input));
                    }
                }
            }
            // map global outputs
            std::transform(globalOutputs.begin(), globalOutputs.end(),
                           topology._connections.begin(),
                           [&](auto const &edge) { return keyToIdx.at(edge); });
            return {
                std::move(topology),
                std::move(nodes),
                std::move(edges),
            };
        }
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_BUILDER_HPP
