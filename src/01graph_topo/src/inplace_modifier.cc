#include "graph_topo/inplace_modifier.h"
#include "common/error_handler.h"
#include "common/range.h"
#include "internal.h"
#include <algorithm>

namespace refactor::graph_topo {

    OnNode::OnNode(size_t idx) noexcept
        : edge(idx) {}
    OnNode OnNode::input(size_t idx) noexcept {
        return OnNode(idx << 1);
    }
    OnNode OnNode::output(size_t idx) noexcept {
        return OnNode((idx << 1) | 1);
    }
    bool OnNode::isInput() const noexcept { return (edge & 1) == 0; }
    bool OnNode::isOutput() const noexcept { return (edge & 1) == 1; }
    size_t OnNode::index() const noexcept { return edge >> 1; }

    InplaceModifier::InplaceModifier(GraphTopo g) noexcept
        : _g(std::move(g)) {}

    auto InplaceModifier::take() noexcept -> GraphTopo {
        return std::move(_g);
    }

    auto InplaceModifier::insert(Bridge bridge) noexcept -> BridgePos {
        auto n = bridge.node;
        auto idx = bridge.edge.index();
        auto &g = *_g._impl;
        auto passConnections = 0ul,
             passEdges = g._globalInputsCount;
        for (auto i : common::range0_(n)) {
            auto const &n_ = g._nodes[i];
            passConnections += n_._inputsCount;
            passEdges += n_._localEdgesCount + n_._outputsCount;
        }

        if (bridge.edge.isInput()) {
            auto currentInput = g._connections[passConnections + idx];
            if (currentInput >= passEdges) {
                // 要替换的输入是局部边
                std::for_each(g._connections.begin() + passConnections, g._connections.end(),
                              [=](auto &c) {
                                  c += c > currentInput ? 1 // 局部边之后的加上了桥的输出
                                       : c >= passEdges ? 2 // 局部边之前的加上了桥的输入和输出
                                                        : 0;// 其他的不变
                              });
                // 当前节点的局部边转移给桥
                --g._nodes[n]._localEdgesCount;
                g._nodes.insert(g._nodes.begin() + n, 1, Node{1, 1, 1});
                // 桥的输入是自己的局部边
                g._connections.insert(g._connections.begin() + passConnections++, 1, passEdges++);
            } else {
                // 要替换的输入是前面节点的输出
                std::for_each(g._connections.begin() + passConnections, g._connections.end(),
                              [=](auto &c) { if (c >= passEdges) { ++c; } });
                // 插入桥和其输入
                g._nodes.insert(g._nodes.begin() + n, 1, Node{0, 1, 1});
                g._connections.insert(g._connections.begin() + passConnections++, 1, currentInput);
            }
            // 当前节点的输入是桥的输出
            g._connections[passConnections + idx] = passEdges;
            return {n, passEdges};
        } else {
            auto const &n_ = g._nodes[n];
            passConnections += n_._inputsCount;
            passEdges += n_._localEdgesCount;
            auto e0 = passEdges + idx;
            auto e1 = passEdges + n_._outputsCount;
            std::for_each(g._connections.begin() + passConnections, g._connections.end(),
                          [=](auto &c) {
                              if (c == e0) {
                                  c = e1;
                              } else if (c >= e1) {
                                  ++c;
                              }
                          });
            g._nodes.insert(g._nodes.begin() + ++n, 1, Node{0, 1, 1});
            g._connections.insert(g._connections.begin() + passConnections, 1, e0);
            return {n, e1};
        }
    }

}// namespace refactor::graph_topo
