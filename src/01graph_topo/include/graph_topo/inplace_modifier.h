#ifndef GRAPH_TOPO_INPLACE_MODIFIER_H
#define GRAPH_TOPO_INPLACE_MODIFIER_H

#include "container.h"

namespace refactor::graph_topo {
    /// @brief 描述连接节点的边。
    class OnNode {
        idx_t edge;

        explicit OnNode(idx_t) noexcept;

    public:
        /// @brief 节点的入边。
        static OnNode input(idx_t) noexcept;
        /// @brief 节点的出边。
        static OnNode output(idx_t) noexcept;
        /// @brief 判断是否为入边。
        bool isInput() const noexcept;
        /// @brief 判断是否为出边。
        bool isOutput() const noexcept;
        /// @brief 获取边的索引。
        idx_t index() const noexcept;
    };

    /// @brief 搭建桥梁节点，即入度出度都为 1 的节点。
    struct Bridge {
        /// @brief 桥接到的节点号。
        idx_t node;
        /// @brief 要桥接的边。
        OnNode edge;
    };

    /// @brief 建好的桥梁节点坐标。
    struct BridgePos {
        /// @brief 桥梁节点和其出边的序号。
        idx_t node, edge;
    };

    class InplaceModifier {
        GraphTopo _g;

    public:
        InplaceModifier() noexcept = default;

        /// @brief 把图拓扑存入修改器。
        explicit InplaceModifier(GraphTopo) noexcept;

        /// @brief 将图拓扑从修改器中取出。
        GraphTopo take() noexcept;

        /// @brief 在图拓扑上搭桥。
        /// @param 桥接描述。
        /// @return 搭好的桥梁节点坐标，与描述对应。
        BridgePos insert(Bridge) noexcept;
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_INPLACE_MODIFIER_H
