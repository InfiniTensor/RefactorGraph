#ifndef GRAPH_TOPO_SEARCHER_HPP
#define GRAPH_TOPO_SEARCHER_HPP

#include "fmtlog.h"
#include "graph_topo.hpp"
#include <algorithm>
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

    /// @brief 构造图拓扑索引时访问边使用这个函数。
    /// @param 边的源节点序号。
    /// @param 边的序号。
    void accessEdge(NodeIdx, EdgeIdx);

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
    Nodes nodes() const;
    /// @brief 获取所有边。
    /// @return 所有边。
    Edges edges() const;
    /// @brief 获取所有全局输入边。
    std::vector<Edge> globalInputs() const;
    /// @brief 获取所有全局输出边。
    std::vector<Edge> globalOutputs() const;

    /// @brief 取出内部的 GraphTopo，并使此 Searcher 对象失效。
    /// @return 内部的 GraphTopo。
    GraphTopo<NodeInfo, EdgeInfo> intoGraphTopo();
};

template<class NodeInfo, class EdgeInfo>
GraphTopoSearcher<NodeInfo, EdgeInfo>::GraphTopoSearcher(
    GraphTopoSearcher<NodeInfo, EdgeInfo>::Internal topo_)
    : topo(std::move(topo_)) {
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
        logd("GraphTopoSearcher/init: node {} with {} output(s)", i, node.edgeCount);
        // 遍历节点的生成的边。
        for (size_t j = 0; j < node.edgeCount; ++j) {
            auto edgeIdx = node.firstEdge.idx + static_cast<idx_t>(j);
            globalInputCandidates.erase({edgeIdx});// 节点生成的边不是全图的输入。
            nodeOutputs[i][j] = {edgeIdx};         // 填写节点输出。
            edgeSource[edgeIdx] = nodeIdx;         // 填写边的源节点。
            logd("GraphTopoSearcher/init: node {} output {} is {}", i, j, edgeIdx);

            accessEdge(nodeIdx, {edgeIdx});
        }
    }
    globalInputs_ = std::vector<EdgeIdx>(globalInputCandidates.begin(), globalInputCandidates.end());
    std::sort(globalInputs_.begin(), globalInputs_.end());
    for (auto const edgeIdx : globalInputs_) {
        accessEdge({-1}, edgeIdx);
        edgeSource[edgeIdx.idx] = {-1};
    }

    // 收缩空间。
    globalInputs_.shrink_to_fit();
    globalOutputs_.shrink_to_fit();
    nodeInputs.shrink_to_fit();
    nodeOutputs.shrink_to_fit();
    nodePredecessors.shrink_to_fit();
    nodeSuccessors.shrink_to_fit();
    edgeSource.shrink_to_fit();
    edgeTargets.shrink_to_fit();

    logi("Built GraphTopoSearcher with {} nodes, {} edges, {} global inputs, {} global outputs.",
         topo.nodes.size(),
         topo.edges.size(),
         globalInputs_.size(),
         globalOutputs_.size());
}

template<class NodeInfo, class EdgeInfo>
void GraphTopoSearcher<NodeInfo, EdgeInfo>::accessEdge(NodeIdx nodeIdx, EdgeIdx edgeIdx) {
    auto const &edge = topo.edges[edgeIdx.idx];
    {// 填写全图输出。
        auto outputIdx = edge.outputIdx.idx;
        if (outputIdx >= 0) {
            if (static_cast<len_t>(outputIdx) >= globalOutputs_.size())
                globalOutputs_.resize(outputIdx + 1);
            globalOutputs_[outputIdx] = edgeIdx;
            logd("GraphTopoSearcher/init: edge {} is global output {}", edgeIdx.idx, outputIdx);
        }
    }

    auto targetIdx = edge.firstTarget;
    while (targetIdx.idx >= 0) {
        auto [next, to] = topo.targets[targetIdx.idx];
        targetIdx = next;

        edgeTargets[edgeIdx.idx].push_back(to);// 填写边的目标节点。
        nodeInputs[to.idx].push_back(edgeIdx); // 填写节点的输入。
        logd("GraphTopoSearcher/init: edge {} has target {}", edgeIdx.idx, to.idx);

        if (nodeIdx.idx >= 0) {
            nodeSuccessors[nodeIdx.idx].insert(to);  // 填写节点的后继。
            nodePredecessors[to.idx].insert(nodeIdx);// 填写节点的前驱。
        }
    }
}

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Nodes {
    GraphTopoSearcher const *graph;

public:
    class Iterator {
        GraphTopoSearcher const *graph;
        idx_t idx;

    public:
        Iterator(GraphTopoSearcher const *graph, idx_t idx) : graph(graph), idx(idx) {}

        bool operator==(Iterator const &rhs) const { return idx == rhs.idx; }
        bool operator!=(Iterator const &rhs) const { return !operator==(rhs); }

        Iterator &operator++() {
            ++idx;
            return *this;
        }

        Node operator*() { return {graph, idx}; }
    };

    Nodes(GraphTopoSearcher const *graph_) : graph(graph_) {}

    Iterator begin() const { return {graph, 0}; }
    Iterator end() const { return {graph, static_cast<idx_t>(size())}; }
    size_t size() const { return graph->topo.nodes.size(); }
    Node operator[](idx_t idx) const {
        ASSERT(0 <= idx && static_cast<len_t>(idx) < size(), "Node index out of range.");
        return {graph, idx};
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Edges {
    GraphTopoSearcher const *graph;

public:
    class Iterator {
        GraphTopoSearcher const *graph;
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

    Edges(GraphTopoSearcher const *graph_) : graph(graph_) {}

    Iterator begin() const { return {graph, 0}; }
    Iterator end() const { return {graph, static_cast<idx_t>(size())}; }
    size_t size() const { return graph->topo.edges.size(); }
    Edge operator[](idx_t idx) const {
        ASSERT(0 <= idx && static_cast<len_t>(idx) < size(), "Edge index out of range.");
        return {graph, idx};
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Node {
    GraphTopoSearcher const *graph;
    idx_t idx;

public:
    Node() : graph(nullptr), idx(-1) {}
    Node(GraphTopoSearcher const *graph_, idx_t idx_) : graph(graph_), idx(idx_) {}
    idx_t index() const { return idx; }
    bool exist() const { return idx >= 0; }
    operator bool() const { return exist(); }
    bool operator==(Node const &rhs) const { return idx == rhs.idx; }
    bool operator!=(Node const &rhs) const { return !operator==(rhs); }
    bool operator<(Node const &rhs) const { return idx < rhs.idx; }
    bool operator>(Node const &rhs) const { return idx > rhs.idx; }
    bool operator<=(Node const &rhs) const { return !operator>(rhs); }
    bool operator>=(Node const &rhs) const { return !operator<(rhs); }
    NodeInfo const &info() const { return graph->topo.nodes[idx].info; }
    std::vector<Edge> inputs() const {
        auto const &inputs = graph->nodeInputs[idx];
        std::vector<Edge> ans(inputs.size());
        std::transform(inputs.begin(), inputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
        return ans;
    }
    std::vector<Edge> outputs() const {
        auto const &outputs = graph->nodeOutputs[idx];
        std::vector<Edge> ans(outputs.size());
        std::transform(outputs.begin(), outputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
        return ans;
    }
    std::set<Node> predecessors() const {
        auto const &predecessors = graph->nodePredecessors[idx];
        std::set<Node> ans;
        std::transform(predecessors.begin(), predecessors.end(), std::inserter(ans, ans.end()),
                       [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
        return ans;
    }
    std::set<Node> successors() const {
        auto const &successors = graph->nodeSuccessors[idx];
        std::set<Node> ans;
        std::transform(successors.begin(), successors.end(), std::inserter(ans, ans.end()),
                       [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
        return ans;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge {
    GraphTopoSearcher const *graph;
    idx_t idx;

public:
    Edge() : graph(nullptr), idx(-1) {}
    Edge(GraphTopoSearcher const *graph, idx_t idx) : graph(graph), idx(idx) {}
    idx_t index() const { return idx; }
    bool exist() const { return idx >= 0; }
    operator bool() const { return exist(); }
    bool operator==(Edge const &rhs) const { return idx == rhs.idx; }
    bool operator!=(Edge const &rhs) const { return !operator==(rhs); }
    bool operator<(Edge const &rhs) const { return idx < rhs.idx; }
    bool operator>(Edge const &rhs) const { return idx > rhs.idx; }
    bool operator<=(Edge const &rhs) const { return !operator>(rhs); }
    bool operator>=(Edge const &rhs) const { return !operator<(rhs); }
    EdgeInfo const &info() const { return graph->topo.edges[idx].info; }
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
typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Nodes
GraphTopoSearcher<NodeInfo, EdgeInfo>::nodes() const { return this; }

template<class NodeInfo, class EdgeInfo>
typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Edges
GraphTopoSearcher<NodeInfo, EdgeInfo>::edges() const { return this; }

template<class NodeInfo, class EdgeInfo>
std::vector<typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge>
GraphTopoSearcher<NodeInfo, EdgeInfo>::globalInputs() const {
    std::vector<Edge> ans(globalInputs_.size());
    std::transform(globalInputs_.begin(), globalInputs_.end(), ans.begin(),
                   [this](auto const &edgeIdx) { return Edge{this, edgeIdx.idx}; });
    return ans;
}

template<class NodeInfo, class EdgeInfo>
std::vector<typename GraphTopoSearcher<NodeInfo, EdgeInfo>::Edge>
GraphTopoSearcher<NodeInfo, EdgeInfo>::globalOutputs() const {
    std::vector<Edge> ans(globalOutputs_.size());
    std::transform(globalOutputs_.begin(), globalOutputs_.end(), ans.begin(),
                   [this](auto const &edgeIdx) { return Edge{this, edgeIdx.idx}; });
    return ans;
}

template<class NodeInfo, class EdgeInfo>
GraphTopo<NodeInfo, EdgeInfo>
GraphTopoSearcher<NodeInfo, EdgeInfo>::intoGraphTopo() {
    return std::move(topo);
}

#endif// GRAPH_TOPO_SEARCHER_HPP
