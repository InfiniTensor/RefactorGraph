#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "graph_topo.hpp"
#include <cstddef>
#include <set>

// 为了方便随便弄个类型占位
// using NodeInfo = int;
// using EdgeInfo = int;

/// @brief 这个类缓存了图拓朴中的各种关系，以支持快速查询。
/// @tparam NodeInfo 节点绑定信息。
/// @tparam EdgeInfo 边绑定信息。
template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher {
    using Internal = GraphTopo<NodeInfo, EdgeInfo>;
    using NodeIdx = typename Internal::NodeIdx;
    using EdgeIdx = typename Internal::EdgeIdx;
    using TargetIdx = typename Internal::TargetIdx;

    std::vector<EdgeIdx> globalInputs, globalOutputs;
    std::vector<std::vector<EdgeIdx>> nodeInputs, nodeOutputs;
    std::vector<std::set<NodeIdx>> nodePredecessors, nodeSuccessors;
    std::vector<NodeIdx> edgeSource;
    std::vector<std::vector<NodeIdx>> edgeTargets;

    void init(Internal const &topo) {
        std::set<EdgeIdx> globalInputCandidates;
        {// 为所有缓存开辟空间。
            auto nodeCount = topo.nodes.size();
            auto edgeCount = topo.edges.size();
            nodeInputs.resize(nodeCount);
            nodeOutputs.resize(nodeCount);
            nodePredecessors.resize(nodeCount);
            nodeSuccessors.resize(nodeCount);
            edgeSource.resize(edgeCount);
            edgeTargets.resize(edgeCount);
            for (size_t i = 0; i < edgeCount; ++i) {
                globalInputCandidates.insert({static_cast<idx_t>(i)});
            }
            globalOutputs.clear();
        }
        // 遍历节点。
        for (size_t i = 0; i < nodeInputs.size(); ++i) {
            auto nodeIdx = NodeIdx{static_cast<idx_t>(i)};
            auto const &node = topo.nodes[i];
            nodeOutputs[i].resize(node.edgeCount);
            // 遍历节点的生成的边。
            for (size_t j = 0; j < node.edgeCount; ++j) {
                auto edgeIdx = node.firstEdge.idx + static_cast<idx_t>(j);
                auto const &edge = topo.edges[edgeIdx];
                auto targetIdx = edge.firstTarget;

                globalInputCandidates.erase({edgeIdx});// 节点生成的边不是全图的输入。
                nodeOutputs[i][j] = {edgeIdx};         // 填写节点输出。
                edgeSource[edgeIdx] = nodeIdx;         // 填写边的源节点。

                {// 填写全图输出。
                    auto outputIdx = edge.outputIdx.idx;
                    if (outputIdx >= 0) {
                        if (outputIdx >= globalOutputs.size())
                            globalOutputs.resize(outputIdx + 1);
                        globalOutputs[outputIdx] = {edgeIdx};
                    }
                }

                while (targetIdx.idx >= 0) {
                    auto [next, to] = topo.targets[targetIdx.idx];
                    targetIdx = next;

                    edgeTargets[edgeIdx].push_back(to);      // 填写边的目标节点。
                    nodeInputs[to.idx].push_back({edgeIdx}); // 填写节点的输入。
                    nodePredecessors[to.idx].insert(nodeIdx);// 填写节点的前驱。
                    nodeSuccessors[i].insert(to);            // 填写节点的后继。
                }
            }
        }
        globalInputs = std::vector<EdgeIdx>(globalInputCandidates.begin(), globalInputCandidates.end());
        std::sort(globalInputs.begin(), globalInputs.end());

        globalInputs.shrink_to_fit();
        globalOutputs.shrink_to_fit();
        nodeInputs.shrink_to_fit();
        nodeOutputs.shrink_to_fit();
        nodePredecessors.shrink_to_fit();
        nodeSuccessors.shrink_to_fit();
        edgeSource.shrink_to_fit();
        edgeTargets.shrink_to_fit();
    }

public:
    explicit GraphTopoSearcher(Internal const &topo) {
        init(topo);
    }

    struct NodeRef;
    struct EdgeRef;

    struct NodeRef {
        GraphTopoSearcher &graph;
        idx_t idx;
    };

    struct EdgeRef {
        GraphTopoSearcher &graph;
        idx_t idx;
    };
};

#endif// GRAPH_TOPO_SEARCHER_HPP
