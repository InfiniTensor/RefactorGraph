#ifndef GRAPH_TOPO_HPP
#define GRAPH_TOPO_HPP

#include <cstdint>
#include <utility>
#include <vector>

using idx_t = int32_t;

// 为了方便随便弄个类型占位
// template<class NodeInfo, class EdgeInfo>
using NodeInfo = int;
using EdgeInfo = int;
struct GraphTopo {
    /// @brief 用于索引节点。
    struct NodeRef {
        idx_t idx;
    };
    /// @brief 用于索引边。
    struct EdgeRef {
        idx_t idx;
    };

    /// @brief 节点。
    struct Node {
        /// @brief 节点信息。
        NodeInfo info;
        /// @brief 节点的第一条边的引用。节点的所有边构成单链表，此为头指针。
        EdgeRef firstEdge;
    };
    /// @brief 边。
    struct Edge {
        /// @brief 边信息。
        EdgeInfo info;
        /// @brief 下一条边的引用。即单链表的下一跳指针。
        EdgeRef next;
        /// @brief 边指向的节点的引用。
        NodeRef to;
        /// @brief 边产生和使用的槽位。
        idx_t inletSlot, outletSlot;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;

    /// @brief 添加节点。
    /// @param info 节点信息。
    /// @return 返回节点的引用，由调用者保存，用于连接边。
    NodeRef addNode(NodeInfo info) {
        NodeRef ans{static_cast<idx_t>(nodes.size())};
        nodes.push_back({std::move(info), EdgeRef{-1}});
        return ans;
    }

    /// @brief 添加边。
    /// @param info 边信息。
    /// @param from 边的起点。
    /// @param to 边的终点。
    void addEdge(EdgeInfo info, NodeRef from, NodeRef to) {
        edges.push_back({
            std::move(info),
            std::exchange(
                nodes[from.idx].firstEdge,
                EdgeRef{static_cast<idx_t>(edges.size())}),
            to,
        });
    }
};

#endif// GRAPH_TOPO_HPP
