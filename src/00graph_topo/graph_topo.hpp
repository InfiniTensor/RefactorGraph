#ifndef GRAPH_TOPO_HPP
#define GRAPH_TOPO_HPP

#include <cstdint>
#include <utility>
#include <vector>

using idx_t = int32_t;
using len_t = std::size_t;

// 为了方便随便弄个类型占位
using NodeInfo = int;
using EdgeInfo = int;

/// @brief 图拓扑表示。
/// @details 图拓扑使用二重的图结构表示。
///          每个节点会产生多个边，节点到边的映射使用前向星表示；
///          每个边会连接到多个节点，边到节点的映射使用链式前向星表示。
///          图拓扑中只包含单向的指向关系，为了获取反向指向关系或其他关系，应基于此结构重新构建。
/// @tparam NodeInfo 节点绑定信息。
/// @tparam EdgeInfo 边绑定信息。
// template<class NodeInfo, class EdgeInfo>

class GraphTopo {
    /// @brief 用于索引节点。
    struct NodeIdx {
        idx_t idx;
    };
    /// @brief 用于索引边。
    struct EdgeIdx {
        idx_t idx;
    };
    /// @brief 用于索引边的目标。
    struct TargetIdx {
        idx_t idx;
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
    /// @brief 用于获取节点的所有输出边。
    class NodeRef {
        NodeIdx idx;
        len_t edgeCount;
    };

    NodeRef addNode(
        NodeInfo info,
        std::vector<EdgeIdx> inputs,
        std::vector<EdgeInfo> outputs) {
        // 添加节点。
        auto nodeIdx = NodeIdx{static_cast<idx_t>(nodes.size())};
        nodes.push_back({
            std::move(info),
            {static_cast<idx_t>(edges.size())},// firstEdge
            static_cast<len_t>(inputs.size()), // edgeCount
        });
        // 将节点加入输入边的目标。
        for (auto edge : inputs) {
            targets.push_back({
                std::exchange(
                    edges[edge.idx].firstTarget,
                    {static_cast<idx_t>(targets.size())}),
                nodeIdx,
            });
        }
        // 添加节点产生的边。
        nodes.reserve(nodes.size() + outputs.size());
        for (auto &edge : outputs) {
            edges.push_back({std::move(edge), {-1}});
        }
        return {nodeIdx, static_cast<len_t>(outputs.size())};
    }
};

#endif// GRAPH_TOPO_HPP
