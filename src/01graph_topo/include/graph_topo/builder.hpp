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
                static_cast<idx_t>(globalOutputs.size()));
            std::vector<Node> nodes;
            std::vector<Edge> edges;

            std::unordered_map<EdgeKey, idx_t> keyToIdx;
            std::unordered_set<EdgeKey> generatedEdges;
            for (auto const &edge : globalInputs) {
                keyToIdx[edge] = static_cast<idx_t>(edges.size());
                if (auto it = this->edges.find(edge); it == this->edges.end()) {
                    edges.emplace_back();
                } else {
                    edges.emplace_back(std::move(it->second));
                }
                generatedEdges.insert(edge);
            }
            for (auto const &[_, value] : this->topology) {
                for (auto const &edge : value.outputs) {
                    generatedEdges.insert(edge);
                }
            }

            std::unordered_set<NodeKey> mappedNodes;
            while (mappedNodes.size() < this->topology.size()) {
                for (auto const &[kn, n__] : this->topology) {
                    auto const &[inputs, outputs] = n__;
                    if (mappedNodes.find(kn) != mappedNodes.end() ||
                        !std::all_of(inputs.begin(), inputs.end(),
                                     [&](auto const &edge) { return keyToIdx.find(edge) != keyToIdx.end() ||
                                                                    generatedEdges.find(edge) == generatedEdges.end(); })) {
                        continue;
                    }
                    mappedNodes.insert(kn);

                    std::unordered_set<EdgeKey> newLocal;
                    std::copy_if(inputs.begin(), inputs.end(), std::inserter(newLocal, newLocal.end()),
                                 [&](auto const &edge) { return keyToIdx.find(edge) == keyToIdx.end() &&
                                                                generatedEdges.find(edge) == generatedEdges.end(); });
                    for (auto const &edge : newLocal) {
                        keyToIdx[edge] = static_cast<idx_t>(edges.size());
                        if (auto it = this->edges.find(edge); it == this->edges.end()) {
                            edges.emplace_back();
                        } else {
                            edges.emplace_back(std::move(it->second));
                        }
                    }
                    nodes.emplace_back(std::move(this->nodes.at(kn)));
                    {
                        topology._nodes.push_back({
                            static_cast<idx_t>(newLocal.size()),
                            static_cast<idx_t>(inputs.size()),
                            static_cast<idx_t>(outputs.size()),
                        });
                        topology._connections.reserve(topology._connections.size() + inputs.size());
                        for (auto const &input : inputs) {
                            topology._connections.push_back(keyToIdx.at(input));
                        }
                    }
                    for (auto const &edge : outputs) {
                        keyToIdx[edge] = static_cast<idx_t>(edges.size());
                        if (auto it = this->edges.find(edge); it == this->edges.end()) {
                            edges.emplace_back();
                        } else {
                            edges.emplace_back(std::move(it->second));
                        }
                    }
                }
            }
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
