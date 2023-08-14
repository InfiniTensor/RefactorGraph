#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "graph_topo.hpp"
#include <cstddef>
#include <set>

/// @brief 这个类缓存了图拓朴中的各种关系，以支持快速查询。
/// @tparam NodeInfo 节点绑定信息。
/// @tparam EdgeInfo 边绑定信息。
template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher {
    using Internal = GraphTopo<NodeInfo, EdgeInfo>;
    using NodeIdx = typename Internal::NodeIdx;
    using EdgeIdx = typename Internal::EdgeIdx;
    using TargetIdx = typename Internal::TargetIdx;

    /// @brief 图拓扑表示。
    Internal topo;
    /// @brief 全局输入边、输出边。
    std::vector<EdgeIdx> globalInputs_, globalOutputs_;
    /// @brief 节点的输入边、输出边。
    std::vector<std::vector<EdgeIdx>> nodeInputs, nodeOutputs;
    /// @brief 节点的前驱节点、后继节点。
    std::vector<std::set<NodeIdx>> nodePredecessors, nodeSuccessors;
    /// @brief 边的源节点。
    std::vector<NodeIdx> edgeSource;
    /// @brief 边的目标节点。
    std::vector<std::vector<NodeIdx>> edgeTargets;

public:
    /// @brief 构造图拓朴索引。
    /// @param topo 图拓朴表示。
    explicit GraphTopoSearcher(Internal topo);

    /// @brief 节点。
    class Node;
    /// @brief 边。
    class Edge;
    /// @brief 所有节点。
    class Nodes;
    /// @brief 所有边。
    class Edges;

    /// @brief 获取所有节点。
    /// @return 所有节点。
    Nodes nodes();
    /// @brief 获取所有边。
    /// @return 所有边。
    Edges edges();
    /// @brief 获取所有全局输入边。
    std::vector<Edge> globalInputs();
    /// @brief 获取所有全局输出边。
    std::vector<Edge> globalOutputs();
};

template<class NodeInfo, class EdgeInfo>
GraphTopoSearcher<NodeInfo, EdgeInfo>::GraphTopoSearcher(
    GraphTopoSearcher<NodeInfo, EdgeInfo>::Internal topo)
    : topo(std::move(topo)) {
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
        globalOutputs_.clear();
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
                    if (outputIdx >= globalOutputs_.size())
                        globalOutputs_.resize(outputIdx + 1);
                    globalOutputs_[outputIdx] = {edgeIdx};
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
    globalInputs_ = std::vector<EdgeIdx>(globalInputCandidates.begin(), globalInputCandidates.end());
    std::sort(globalInputs_.begin(), globalInputs_.end());

    // 收缩空间。
    globalInputs_.shrink_to_fit();
    globalOutputs_.shrink_to_fit();
    nodeInputs.shrink_to_fit();
    nodeOutputs.shrink_to_fit();
    nodePredecessors.shrink_to_fit();
    nodeSuccessors.shrink_to_fit();
    edgeSource.shrink_to_fit();
    edgeTargets.shrink_to_fit();
}

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Nodes {
    GraphTopoSearcher *graph;

public:
    class Iterator {
        GraphTopoSearcher *graph;
        idx_t idx;

    public:
        Iterator(GraphTopoSearcher *graph, idx_t idx) : graph(graph), idx(idx) {}

        bool operator==(Iterator const &rhs) const { return idx == rhs.idx; }
        bool operator!=(Iterator const &rhs) const { return !operator==(rhs); }

        Iterator &operator++() {
            ++idx;
            return *this;
        }

        Node operator*() { return {graph, idx}; }
    };

    Nodes(GraphTopoSearcher *graph) : graph(graph) {}

    Iterator begin() const { return {graph, 0}; }
    Iterator end() const { return {graph, static_cast<idx_t>(graph->nodeInputs.size())}; }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Edges {
    GraphTopoSearcher *graph;

public:
    class Iterator {
        GraphTopoSearcher *graph;
        idx_t idx;

    public:
        Iterator(GraphTopoSearcher const *graph, idx_t idx) : graph(graph), idx(idx) {}

        bool operator==(Iterator const &rhs) const { return idx == rhs.idx; }
        bool operator!=(Iterator const &rhs) const { return !operator==(rhs); }

        Iterator &operator++() {
            ++idx;
            return *this;
        }

        Edge operator*() const { return {graph, idx}; }
    };

    Edges(GraphTopoSearcher const *graph) : graph(graph) {}

    Iterator begin() const { return {graph, 0}; }
    Iterator end() const { return {graph, static_cast<idx_t>(graph->edgeSource.size())}; }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Node {
    GraphTopoSearcher *graph;
    idx_t idx;

public:
    Node() : graph(nullptr), idx(-1) {}
    Node(GraphTopoSearcher *graph, idx_t idx) : graph(graph), idx(idx) {}
    NodeInfo &info() { return graph->topo.nodes[idx].info; }
    std::vector<Edge> inputs() {
        auto const &inputs = graph->nodeInputs[idx];
        std::vector<Edge> ans(inputs.size());
        std::transform(inputs.begin(), inputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
        return ans;
    }
    std::vector<Edge> outputs() {
        auto const &outputs = graph->nodeOutputs[idx];
        std::vector<Edge> ans(outputs.size());
        std::transform(outputs.begin(), outputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
        return ans;
    }
    std::set<Node> predecessors() {
        auto const &predecessors = graph->nodePredecessors[idx];
        std::set<Node> ans;
        std::transform(predecessors.begin(), predecessors.end(), std::inserter(ans, ans.end()),
                       [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
        return ans;
    }
    std::set<Node> successors() {
        auto const &successors = graph->nodeSuccessors[idx];
        std::set<Node> ans;
        std::transform(successors.begin(), successors.end(), std::inserter(ans, ans.end()),
                       [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
        return ans;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge {
    GraphTopoSearcher *graph;
    idx_t idx;

public:
    Edge() : graph(nullptr), idx(-1) {}
    Edge(GraphTopoSearcher *graph, idx_t idx) : graph(graph), idx(idx) {}
    EdgeInfo &info() { return graph->topo.edges[idx].info; }
    Node source() { return Node{graph, graph->edgeSource[idx].idx}; }
    std::vector<Node> targets() {
        auto const &targets = graph->edgeTargets[idx];
        std::vector<Node> ans(targets.size());
        std::transform(targets.begin(), targets.end(), ans.begin(),
                       [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
        return ans;
    }
};

template<class NodeInfo, class EdgeInfo>
GraphTopoSearcher<NodeInfo, EdgeInfo>::Nodes GraphTopoSearcher<NodeInfo, EdgeInfo>::nodes() { return this; }
template<class NodeInfo, class EdgeInfo>
GraphTopoSearcher<NodeInfo, EdgeInfo>::Edges GraphTopoSearcher<NodeInfo, EdgeInfo>::edges() { return this; }
template<class NodeInfo, class EdgeInfo>
std::vector<typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge> GraphTopoSearcher<NodeInfo, EdgeInfo>::globalInputs() {
    std::vector<Edge> ans(globalInputs_.size());
    std::transform(globalInputs_.begin(), globalInputs_.end(), ans.begin(),
                   [this](auto const &edgeIdx) { return Edge{*this, edgeIdx.idx}; });
    return ans;
}
template<class NodeInfo, class EdgeInfo>
std::vector<typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge> GraphTopoSearcher<NodeInfo, EdgeInfo>::globalOutputs() {
    std::vector<Edge> ans(globalOutputs_.size());
    std::transform(globalOutputs_.begin(), globalOutputs_.end(), ans.begin(),
                   [this](auto const &edgeIdx) { return Edge{*this, edgeIdx.idx}; });
    return ans;
}

#endif// GRAPH_TOPO_SEARCHER_HPP
