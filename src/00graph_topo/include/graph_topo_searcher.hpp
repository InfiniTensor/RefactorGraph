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

    Internal topo;
    std::vector<EdgeIdx> globalInputs_, globalOutputs_;
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

        globalInputs_.shrink_to_fit();
        globalOutputs_.shrink_to_fit();
        nodeInputs.shrink_to_fit();
        nodeOutputs.shrink_to_fit();
        nodePredecessors.shrink_to_fit();
        nodeSuccessors.shrink_to_fit();
        edgeSource.shrink_to_fit();
        edgeTargets.shrink_to_fit();
    }

public:
    explicit GraphTopoSearcher(Internal topo) : topo(std::move(topo)) {
        init(topo);
    }

    class Node;
    class Edge;
    class Nodes;
    class Edges;

    class Nodes {
        GraphTopoSearcher &graph;

    public:
        class Iterator {
            GraphTopoSearcher &graph;
            idx_t idx;

        public:
            Iterator(GraphTopoSearcher &graph, idx_t idx) : graph(graph), idx(idx) {}

            bool operator==(Iterator const &rhs) const { return idx == rhs.idx; }
            bool operator!=(Iterator const &rhs) const { return !operator==(rhs); }

            Iterator &operator++() {
                ++idx;
                return *this;
            }

            Node operator*() { return {graph, idx}; }
        };

        Nodes(GraphTopoSearcher &graph) : graph(graph) {}

        Iterator begin() const { return {graph, 0}; }
        Iterator end() const { return {graph, static_cast<idx_t>(graph.nodeInputs.size())}; }
    };

    class Edges {
        GraphTopoSearcher &graph;

    public:
        class Iterator {
            GraphTopoSearcher &graph;
            idx_t idx;

        public:
            Iterator(GraphTopoSearcher const &graph, idx_t idx) : graph(graph), idx(idx) {}

            bool operator==(Iterator const &rhs) const { return idx == rhs.idx; }
            bool operator!=(Iterator const &rhs) const { return !operator==(rhs); }

            Iterator &operator++() {
                ++idx;
                return *this;
            }

            Edge operator*() const { return {graph, idx}; }
        };

        Edges(GraphTopoSearcher const &graph) : graph(graph) {}

        Iterator begin() const { return {graph, 0}; }
        Iterator end() const { return {graph, static_cast<idx_t>(graph.edgeSource.size())}; }
    };

    class Node {
        GraphTopoSearcher &graph;
        idx_t idx;

    public:
        Node(GraphTopoSearcher &graph, idx_t idx) : graph(graph), idx(idx) {}
        NodeInfo &info() { return graph.topo.nodes[idx].info; }
        std::vector<Edge> inputs() {
            auto const &inputs = graph.nodeInputs[idx];
            std::vector<Edge> ans(inputs.size());
            std::transform(inputs.begin(), inputs.end(), ans.begin(),
                           [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
            return ans;
        }
        std::vector<Edge> outputs() {
            auto const &outputs = graph.nodeOutputs[idx];
            std::vector<Edge> ans(outputs.size());
            std::transform(outputs.begin(), outputs.end(), ans.begin(),
                           [this](auto const &edgeIdx) { return Edge{graph, edgeIdx.idx}; });
            return ans;
        }
        std::set<Node> predecessors() {
            auto const &predecessors = graph.nodePredecessors[idx];
            std::set<Node> ans;
            std::transform(predecessors.begin(), predecessors.end(), std::inserter(ans, ans.end()),
                           [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
            return ans;
        }
        std::set<Node> successors() {
            auto const &successors = graph.nodeSuccessors[idx];
            std::set<Node> ans;
            std::transform(successors.begin(), successors.end(), std::inserter(ans, ans.end()),
                           [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
            return ans;
        }
    };

    class Edge {
        GraphTopoSearcher &graph;
        idx_t idx;

    public:
        Edge(GraphTopoSearcher &graph, idx_t idx) : graph(graph), idx(idx) {}
        EdgeInfo &info() { return graph.topo.edges[idx].info; }
        Node source() { return Node{graph, graph.edgeSource[idx].idx}; }
        std::vector<Node> targets() {
            auto const &targets = graph.edgeTargets[idx];
            std::vector<Node> ans(targets.size());
            std::transform(targets.begin(), targets.end(), ans.begin(),
                           [this](auto const &nodeIdx) { return Node{graph, nodeIdx.idx}; });
            return ans;
        }
    };

    Nodes nodes() { return *this; }
    Edges edges() { return *this; }
    std::vector<Edge> globalInputs() {
        std::vector<Edge> ans(globalInputs_.size());
        std::transform(globalInputs_.begin(), globalInputs_.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{*this, edgeIdx.idx}; });
        return ans;
    }
    std::vector<Edge> globalOutputs() {
        std::vector<Edge> ans(globalOutputs_.size());
        std::transform(globalOutputs_.begin(), globalOutputs_.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return Edge{*this, edgeIdx.idx}; });
        return ans;
    }
};

#endif// GRAPH_TOPO_SEARCHER_HPP
