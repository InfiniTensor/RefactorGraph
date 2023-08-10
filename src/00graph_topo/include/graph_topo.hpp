#ifndef GRAPH_TOPO_HPP
#define GRAPH_TOPO_HPP

#include <cassert>
#include <cstdint>
#include <utility>
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

    /// @brief 用于索引节点。
    struct NodeIdx {
        idx_t idx;

        bool operator<(NodeIdx const &rhs) const {
            return idx < rhs.idx;
        }
    };
    /// @brief 用于索引边。
    struct EdgeIdx {
        idx_t idx;

        bool operator<(EdgeIdx const &rhs) const {
            return idx < rhs.idx;
        }
    };
    /// @brief 用于索引边的目标。
    struct TargetIdx {
        idx_t idx;

        bool operator<(TargetIdx const &rhs) const {
            return idx < rhs.idx;
        }
    };

    /// @brief 节点。
    struct Node {
        /// @brief 节点信息。
        NodeInfo info;
        /// @brief 节点的第一条边的引用。节点的所有边构成单链表，此为头指针。
        EdgeIdx firstEdge;
        /// @brief 节点产生的边的数量。
        len_t edgeCount;
    };
    /// @brief 边。
    struct Edge {
        /// @brief 边信息。
        EdgeInfo info;
        /// @brief 下一个目标的引用。边的所有目标构成单链表，此为头指针。
        TargetIdx firstTarget;
    };
    /// @brief 边的目标。
    struct Target {
        /// @brief 下一个目标的引用。
        TargetIdx next;
        /// @brief 指向目标节点。
        NodeIdx to;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Target> targets;

public:
    class NodeRef;
    class EdgeRef;
    class NodeProductRef;

    /// @brief 图中一个节点的引用。
    class NodeRef {
        friend GraphTopo;
        friend NodeProductRef;
        NodeIdx idx;
        NodeRef(idx_t idx) : idx(idx) {}
    };

    /// @brief 图中一条边的引用。
    class EdgeRef {
        friend GraphTopo;
        friend NodeProductRef;
        EdgeIdx idx;
        EdgeRef(idx_t idx) : idx(idx) {}
    };

    class NodeProductRef {
        friend GraphTopo;
        friend EdgeRef;
        NodeIdx idx;
        EdgeIdx firstEdge;
        len_t edgeCount;

        NodeProductRef(NodeIdx idx, EdgeIdx firstEdge, len_t edgeCount)
            : idx(idx), firstEdge(firstEdge), edgeCount(edgeCount) {}

    public:
        NodeRef node() const {
            return NodeRef(idx);
        }

        EdgeRef operator[](idx_t i) const {
            if (i < 0 || edgeCount <= i) {
                throw std::out_of_range("Edge index out of range");
            }
            return EdgeRef(firstEdge.idx + i);
        }
    };

    /// @brief 添加边。
    /// @param info 边信息。
    /// @return 边的引用。
    EdgeRef addEdge(EdgeInfo info) {
        edges.push_back({std::move(info), {-1}});
        return EdgeRef{static_cast<idx_t>(edges.size()) - 1};
    }

    /// @brief 添加节点。
    /// @param info 节点信息。
    /// @param inputs 输入。
    /// @param outputs 输出。
    /// @return 节点的引用。
    NodeProductRef addNode(
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
        nodes.reserve(nodes.size() + outputs.size());
        for (auto &edge : outputs) {
            edges.push_back({std::move(edge), {-1}});
        }
        return NodeProductRef(nodeIdx, firstEdge, edgeCount);
    }

    /// @brief 获取节点信息。
    /// @param ref 节点在图上的引用。
    /// @return 节点信息。
    NodeInfo &getInfo(NodeRef ref) {
        return nodes[ref.idx.idx].info;
    }

    /// @brief 获取边信息。
    /// @param ref 边在图上的引用。
    /// @return 边信息。
    EdgeInfo &getInfo(EdgeRef ref) {
        return edges[ref.idx.idx].info;
    }
};

#endif// GRAPH_TOPO_HPP
