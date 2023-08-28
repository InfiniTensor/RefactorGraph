#ifndef GRAPH_TOPO_HPP
#define GRAPH_TOPO_HPP

#include "common/error_handler.h"
#include <algorithm>
#include <cstdint>
#include <vector>

using idx_t = int32_t;
using len_t = std::size_t;

template<class NodeInfo, class EdgeInfo>
class GraphTopoSearcher;

/// @brief 图拓扑表示。
/// @details 图拓扑使用二重的图结构表示。
///          每个节点会产生多个边，节点到边的映射使用前向星表示；
///          每个边会连接到多个节点，边到节点的映射使用链式前向星表示。
///          图拓扑中只包含单向的指向关系，为了获取反向指向关系或其他关系，应基于此结构重新构建。
/// @tparam NodeInfo 节点绑定信息。
/// @tparam EdgeInfo 边绑定信息。
template<class NodeInfo, class EdgeInfo>
class GraphTopo {
    friend class GraphTopoSearcher<NodeInfo, EdgeInfo>;

    template<class OtherNodeInfo, class OtherEdgeInfo>
    friend class GraphTopo;

    /// @brief 用于索引节点。
    class NodeIdx;
    /// @brief 用于索引边。
    class EdgeIdx;
    /// @brief 用于索引边的目标。
    class TargetIdx;
    /// @brief 用于索引边作为全局输出的顺序。
    class OutputIdx;

    /// @brief 节点。
    class Node;
    /// @brief 边。
    class Edge;
    /// @brief 边的目标。
    class Target;

    /// @brief 保存所有节点。
    std::vector<Node> nodes;
    /// @brief 保存所有边。
    std::vector<Edge> edges;
    /// @brief 保存所有边的目标。
    std::vector<Target> targets;

public:
    /// @brief 图中一个节点的引用。
    class NodeRef;
    /// @brief 图中一条边的引用。
    class EdgeRef;
    /// @brief 节点的产品（输出边）。
    class NodeProduct;

    /// @brief 添加全局输入边。
    /// @param info 边信息。
    /// @return 边的引用。
    EdgeRef addEdge(EdgeInfo info);
    /// @brief 标记全局输出边。
    /// @param globalOutputs 所有全局输出边的列表。
    void markOutput(EdgeRef globalOutput);
    /// @brief 标记全局输出边。
    /// @param globalOutputs 所有全局输出边的列表。
    void markOutput(std::vector<EdgeRef> const &globalOutputs);
    /// @brief 添加节点。
    /// @param info 节点信息。
    /// @param inputs 输入。
    /// @param outputs 输出。
    /// @return 节点的引用。
    NodeProduct addNode(NodeInfo info, std::vector<EdgeRef> inputs, std::vector<EdgeInfo> outputs);
    /// @brief 获取节点信息。
    /// @param ref 节点在图上的引用。
    /// @return 节点信息。
    NodeInfo &getInfo(NodeRef ref);
    /// @brief 获取边信息。
    /// @param ref 边在图上的引用。
    /// @return 边信息。
    EdgeInfo &getInfo(EdgeRef ref);
    /// @brief 在不改变拓扑结构的情况下映射节点和边信息。这个函数将导致当前图拓扑失效。
    /// @tparam NewNodeInfo 新的节点信息类型。
    /// @tparam NewEdgeInfo 新的边信息类型。
    /// @param _1 映射节点信息。
    /// @param _2 映射边信息。
    /// @return 映射后的新图拓扑。
    template<class NewNodeInfo, class NewEdgeInfo>
    GraphTopo<NewNodeInfo, NewEdgeInfo> map(NewNodeInfo(NodeInfo &&), NewEdgeInfo(EdgeInfo &&));
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::NodeIdx {
public:
    idx_t idx;

    bool operator<(NodeIdx const &rhs) const {
        return idx < rhs.idx;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::EdgeIdx {
public:
    idx_t idx;

    bool operator<(EdgeIdx const &rhs) const {
        return idx < rhs.idx;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::TargetIdx {
public:
    idx_t idx;

    bool operator<(TargetIdx const &rhs) const {
        return idx < rhs.idx;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::OutputIdx {
public:
    idx_t idx;

    bool operator<(OutputIdx const &rhs) const {
        return idx < rhs.idx;
    }
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::Node {
public:
    /// @brief 节点信息。
    NodeInfo info;
    /// @brief 节点的第一条边的引用。节点的所有边构成单链表，此为头指针。
    EdgeIdx firstEdge;
    /// @brief 节点产生的边的数量。
    len_t edgeCount;
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::Edge {
public:
    /// @brief 边信息。
    EdgeInfo info;
    /// @brief 下一个目标的引用。边的所有目标构成单链表，此为头指针。
    TargetIdx firstTarget;
    /// @brief 如果边是全局输出的话，标记边作为输出的序号。
    OutputIdx outputIdx;
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::Target {
public:
    /// @brief 下一个目标的引用。
    TargetIdx next;
    /// @brief 指向目标节点。
    NodeIdx to;
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::NodeRef {
    friend GraphTopo;
    friend NodeProduct;
    NodeIdx idx;
    NodeRef(idx_t idx_) : idx({idx_}) {}

public:
    NodeRef() : NodeRef(-1) {}
};

/// @brief 图中一条边的引用。
template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::EdgeRef {
    friend GraphTopo;
    friend NodeProduct;
    EdgeIdx idx;
    EdgeRef(idx_t idx_) : idx({idx_}) {}

public:
    EdgeRef() : EdgeRef(-1) {}
};

template<class NodeInfo, class EdgeInfo>
class GraphTopo<NodeInfo, EdgeInfo>::NodeProduct {
    friend GraphTopo;
    friend EdgeRef;
    NodeIdx idx;
    EdgeIdx firstEdge;
    len_t edgeCount;

    NodeProduct(NodeIdx idx_, EdgeIdx firstEdge_, len_t edgeCount_)
        : idx(idx_), firstEdge(firstEdge_), edgeCount(edgeCount_) {}

public:
    NodeRef node() const {
        return NodeRef(idx);
    }

    EdgeRef operator[](idx_t i) const {
        if (i < 0 || edgeCount <= static_cast<len_t>(i)) {
            OUT_OF_RANGE("Edge index out of range", i, edgeCount);
        }
        return EdgeRef(firstEdge.idx + i);
    }
};

template<class NodeInfo, class EdgeInfo>
typename GraphTopo<NodeInfo, EdgeInfo>::EdgeRef
GraphTopo<NodeInfo, EdgeInfo>::addEdge(EdgeInfo info) {
    edges.push_back({std::move(info), {-1}, {-1}});
    return EdgeRef(static_cast<idx_t>(edges.size()) - 1);
}

/// @brief 标记全局输出边。
/// @param globalOutputs 所有全局输出边的列表。
template<class NodeInfo, class EdgeInfo>
void GraphTopo<NodeInfo, EdgeInfo>::markOutput(EdgeRef globalOutput) {
    ASSERT(0 <= globalOutput.idx.idx && globalOutput.idx.idx < static_cast<idx_t>(edges.size()),
           "Edge index out of range");
    auto max = std::max_element(
                   edges.begin(),
                   edges.end(),
                   [](auto const &a, auto const &b) { return a.outputIdx.idx < b.outputIdx.idx; })
                   ->outputIdx.idx;
    edges[globalOutput.idx.idx].outputIdx = {++max};
}

/// @brief 标记全局输出边。
/// @param globalOutputs 所有全局输出边的列表。
template<class NodeInfo, class EdgeInfo>
void GraphTopo<NodeInfo, EdgeInfo>::markOutput(std::vector<EdgeRef> const &globalOutputs) {
    ASSERT(std::all_of(globalOutputs.begin(), globalOutputs.end(),
                       [this](auto ref) { return 0 <= ref.idx.idx && ref.idx.idx < static_cast<idx_t>(edges.size()); }),
           "Edge index out of range");

    auto max = std::max_element(
                   edges.begin(),
                   edges.end(),
                   [](auto const &a, auto const &b) { return a.outputIdx.idx < b.outputIdx.idx; })
                   ->outputIdx.idx;
    for (auto const &globalOutput : globalOutputs) {
        edges[globalOutput.idx.idx].outputIdx = {++max};
    }
}

template<class NodeInfo, class EdgeInfo>
typename GraphTopo<NodeInfo, EdgeInfo>::NodeProduct
GraphTopo<NodeInfo, EdgeInfo>::addNode(
    NodeInfo info,
    std::vector<EdgeRef> inputs,
    std::vector<EdgeInfo> outputs) {
    auto nodeIdx = NodeIdx{static_cast<idx_t>(nodes.size())};
    auto firstEdge = EdgeIdx{static_cast<idx_t>(edges.size())};
    auto edgeCount = static_cast<len_t>(outputs.size());
    // 添加节点。
    nodes.push_back({std::move(info), firstEdge, edgeCount});
    // 将节点加入输入边的目标。
    for (auto edge : inputs) {
        targets.push_back({
            std::exchange(
                edges[edge.idx.idx].firstTarget,
                {static_cast<idx_t>(targets.size())}),
            nodeIdx,
        });
    }
    // 添加节点产生的边。
    edges.reserve(edges.size() + outputs.size());
    for (auto &edge : outputs) {
        edges.push_back({std::move(edge), {-1}, {-1}});
    }
    return NodeProduct(nodeIdx, firstEdge, edgeCount);
}

template<class NodeInfo, class EdgeInfo>
NodeInfo &
GraphTopo<NodeInfo, EdgeInfo>::getInfo(NodeRef ref) {
    return nodes[ref.idx.idx].info;
}

template<class NodeInfo, class EdgeInfo>
EdgeInfo &
GraphTopo<NodeInfo, EdgeInfo>::getInfo(EdgeRef ref) {
    return edges[ref.idx.idx].info;
}

template<class NodeInfo, class EdgeInfo>
template<class NewNodeInfo, class NewEdgeInfo>
GraphTopo<NewNodeInfo, NewEdgeInfo> GraphTopo<NodeInfo, EdgeInfo>::map(
    NewNodeInfo mapNode(NodeInfo &&),
    NewEdgeInfo mapEdge(EdgeInfo &&)) {

    GraphTopo<NewNodeInfo, NewEdgeInfo> ans;
    ans.nodes.reserve(nodes.size());
    ans.edges.reserve(edges.size());
    ans.targets.reserve(targets.size());
    for (auto &node : nodes) {
        ans.nodes.push_back({mapNode(std::move(node.info)), {node.firstEdge.idx}, node.edgeCount});
    }
    for (auto &edge : edges) {
        ans.edges.push_back({mapEdge(std::move(edge.info)), {edge.firstTarget.idx}, {edge.outputIdx.idx}});
    }
    for (auto &target : targets) {
        ans.targets.push_back({{target.next.idx}, {target.to.idx}});
    }
    return ans;
}

#endif// GRAPH_TOPO_HPP
