#ifndef GRAPH_TOPO_BUILDER_HPP
#define GRAPH_TOPO_BUILDER_HPP

#include "graph_topo.hpp"

template<class NodeInfo, class EdgeInfo, class EdgeKey>
class GraphTopoBuilder {
    GraphTopo<NodeInfo, EdgeInfo> graph;

public:
    void addEdge(EdgeKey);
    void addNode(
        NodeInfo info,
        std::vector<EdgeKey> const &inputs,
        std::vector<EdgeKey> const &outputs);
    GraphTopo<NodeInfo, EdgeInfo> build() {
        auto ans = std::move(graph);

        // TODO

        return ans;
    }
};

#endif// GRAPH_TOPO_BUILDER_HPP
