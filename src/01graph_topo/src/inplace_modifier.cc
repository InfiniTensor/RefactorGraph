#include "graph_topo/inplace_modifier.h"

namespace refactor::graph_topo {

    OnNode::OnNode(count_t idx) noexcept
        : edge(idx) {}
    OnNode OnNode::input(count_t idx) noexcept {
        return OnNode(idx << 1);
    }
    OnNode OnNode::output(count_t idx) noexcept {
        return OnNode((idx << 1) | 1);
    }
    bool OnNode::isInput() const noexcept { return (edge & 1) == 0; }
    bool OnNode::isOutput() const noexcept { return (edge & 1) == 1; }
    count_t OnNode::index() const noexcept { return edge >> 1; }

    InplaceModifier::InplaceModifier(GraphTopo _g) noexcept
        : _g(std::move(_g)) {}

    auto InplaceModifier::take() noexcept -> GraphTopo {
        return std::move(_g);
    }

    auto InplaceModifier::insert(Bridge bridge) noexcept -> BridgePos {
        auto n = bridge.node;
        auto idx = bridge.edge.index();
        auto passConnections = _g._lenOut,
             passEdges = _g._lenIn;
        for (auto i : range0_(n)) {
            auto const &n_ = _g._nodes[i];
            passConnections += n_._inputsCount;
            passEdges += n_._localEdgesCount + n_._outputsCount;
        }

        if (bridge.edge.isInput()) {
            auto currentInput = _g._connections[passConnections + idx];
            if (currentInput >= passEdges) {
                // 要替换的输入是局部边
                std::for_each(_g._connections.begin() + passConnections, _g._connections.end(),
                              [=](auto &c) {
                                  c += c > currentInput ? 1 // 局部边之后的加上了桥的输出
                                       : c >= passEdges ? 2 // 局部边之前的加上了桥的输入和输出
                                                        : 0;// 其他的不变
                              });
                // 当前节点的局部边转移给桥
                --_g._nodes[n]._localEdgesCount;
                _g._nodes.insert(_g._nodes.begin() + n, 1, GraphTopo::Node{1, 1, 1});
                // 桥的输入是自己的局部边
                _g._connections.insert(_g._connections.begin() + passConnections++, 1, passEdges++);
            } else {
                // 要替换的输入是前面节点的输出
                std::for_each(_g._connections.begin() + passConnections, _g._connections.end(),
                              [=](auto &c) { if (c >= passEdges) { ++c; } });
                // 插入桥和其输入
                _g._nodes.insert(_g._nodes.begin() + n, 1, GraphTopo::Node{0, 1, 1});
                _g._connections.insert(_g._connections.begin() + passConnections++, 1, currentInput);
            }
            // 当前节点的输入是桥的输出
            _g._connections[passConnections + idx] = passEdges;
            return {n, passEdges};
        } else {
            auto const &n_ = _g._nodes[n];
            passConnections += n_._inputsCount;
            passEdges += n_._localEdgesCount;
            auto e0 = passEdges + idx;
            auto e1 = passEdges + n_._outputsCount;
            std::for_each(_g._connections.begin() + passConnections, _g._connections.end(),
                          [=](auto &c) {
                              if (c == e0) {
                                  c = e1;
                              } else if (c >= e1) {
                                  ++c;
                              }
                          });
            _g._nodes.insert(_g._nodes.begin() + ++n, 1, GraphTopo::Node{0, 1, 1});
            _g._connections.insert(_g._connections.begin() + passConnections, 1, e0);
            return {n, e1};
        }
    }

    auto InplaceModifier::reconnect(count_t from, count_t to) -> size_t {
        size_t ans = 0;
        for (auto &c : _g._connections) {
            if (c == from) {
                c = to;
                ++ans;
            }
        }
        return ans;
    }

    auto InplaceModifier::reconnect(std::unordered_map<count_t, count_t> const &map) -> size_t {
        size_t ans = 0;
        for (auto &c : _g._connections) {
            if (auto it = map.find(c); it != map.end()) {
                c = it->second;
                ++ans;
            }
        }
        return ans;
    }

}// namespace refactor::graph_topo
