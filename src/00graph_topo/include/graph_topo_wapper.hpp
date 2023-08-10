#ifndef GRAPH_TOPO_WAPPER_HPP
#define GRAPH_TOPO_WAPPER_HPP

#include "graph_topo.hpp"
#include <cstddef>
#include <set>

// 为了方便随便弄个类型占位
using NodeInfo = int;
using EdgeInfo = int;

// template<class NodeInfo, class EdgeInfo>
class GraphTopoWapper {
    using Internal = GraphTopo<NodeInfo, EdgeInfo>;
    Internal internal;

    std::set<Internal::EdgeIdx> globalInputs, globalOutputs;
    std::vector<std::vector<Internal::EdgeIdx>> nodeInputs, nodeOutputs;
    std::vector<std::set<Internal::NodeIdx>> nodePredecessors, nodeSuccessors;
    std::vector<Internal::NodeIdx> edgeSource;
    std::vector<std::vector<Internal::NodeIdx>> edgeTargets;

    void init() {
        {// 为所有缓存开辟空间。
            auto nodeCount = internal.nodes.size();
            auto edgeCount = internal.edges.size();
            nodeInputs.resize(nodeCount);
            nodeOutputs.resize(nodeCount);
            nodePredecessors.resize(nodeCount);
            nodeSuccessors.resize(nodeCount);
            edgeSource.resize(edgeCount);
            edgeTargets.resize(edgeCount);
            for (size_t i = 0; i < edgeCount; ++i) {
                globalInputs.insert({static_cast<idx_t>(i)});
            }
            globalOutputs.clear();
        }
        // 遍历节点。
        for (size_t i = 0; i < nodeInputs.size(); ++i) {
            auto nodeIdx = Internal::NodeIdx{static_cast<idx_t>(i)};
            auto const &node = internal.nodes[i];
            nodeOutputs[i].resize(node.edgeCount);
            // 遍历节点的生成的边。
            for (size_t j = 0; j < node.edgeCount; ++j) {
                auto edgeIdx = node.firstEdge.idx + static_cast<idx_t>(j);
                auto const &edge = internal.edges[edgeIdx];
                auto targetIdx = edge.firstTarget;

                globalInputs.erase({edgeIdx});// 节点生成的边不是全图的输入。
                nodeOutputs[i][j] = {edgeIdx};// 填写节点输出。
                edgeSource[edgeIdx] = nodeIdx;// 填写边的源节点。
                if (targetIdx.idx < 0) {
                    globalOutputs.insert({edgeIdx});// 没有目标的边是全图的输出。
                } else {
                    do {
                        auto [next, to] = internal.targets[targetIdx.idx];
                        targetIdx = next;

                        edgeTargets[edgeIdx].push_back(to);      // 填写边的目标节点。
                        nodeInputs[to.idx].push_back({edgeIdx}); // 填写节点的输入。
                        nodePredecessors[to.idx].insert(nodeIdx);// 填写节点的前驱。
                        nodeSuccessors[i].insert(to);            // 填写节点的后继。
                    } while (targetIdx.idx >= 0);
                }
            }
        }
    }

public:
    struct NodeRef;
    struct EdgeRef;

    struct NodeRef {
        GraphTopoWapper &graph;
        idx_t idx;
    };

    struct EdgeRef {
        GraphTopoWapper &graph;
        idx_t idx;
    };
};

#endif// GRAPH_TOPO_WAPPER_HPP
