#ifndef GRAPH_TOPO_BUILDER_HPP
#define GRAPH_TOPO_BUILDER_HPP

#include "container.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace refactor::graph_topo {

    template<class Node, class Edge>
    struct Graph {
        GraphTopo topology;
        std::vector<Node> nodes;
        std::vector<Edge> edges;
    };

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
            auto topology = GraphTopo::__withGlobalInputs(globalInputs.size());
            std::vector<Node> nodes;
            std::vector<Edge> edges;

            std::unordered_map<EdgeKey, size_t> keyToIdx;
            std::unordered_set<EdgeKey> generatedEdges;
            for (auto const &edge : globalInputs) {
                keyToIdx.insert({edge, edges.size()});
                auto it = this->edges.find(edge);
                if (it == this->edges.end()) {
                    edges.push_back({});
                } else {
                    edges.push_back(std::move(it->second));
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
                        keyToIdx.insert({edge, edges.size()});
                        auto it = this->edges.find(edge);
                        if (it == this->edges.end()) {
                            edges.push_back({});
                        } else {
                            edges.push_back(std::move(it->second));
                        }
                    }
                    nodes.push_back(std::move(this->nodes.at(kn)));
                    {
                        std::vector<size_t> nodeInputs(inputs.size());
                        std::transform(inputs.begin(), inputs.end(), nodeInputs.begin(),
                                       [&](auto const &edge) { return keyToIdx.at(edge); });
                        topology.__addNode(newLocal.size(), std::move(nodeInputs), outputs.size());
                    }
                    for (auto const &edge : outputs) {
                        keyToIdx.insert({edge, edges.size()});
                        auto it = this->edges.find(edge);
                        if (it == this->edges.end()) {
                            edges.push_back({});
                        } else {
                            edges.push_back(std::move(it->second));
                        }
                    }
                }
            }
            {
                std::vector<size_t> globalOutputs;
                std::transform(this->globalOutputs.begin(), this->globalOutputs.end(), std::back_inserter(globalOutputs),
                               [&](auto const &edge) { return keyToIdx.at(edge); });
                topology.__setGlobalOutputs(std::move(globalOutputs));
            }

            return {
                std::move(topology),
                std::move(nodes),
                std::move(edges),
            };
        }
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_BUILDER_HPP
