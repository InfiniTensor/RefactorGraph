#include "graph_topo/modifier.h"
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

    Modifier::Modifier(GraphTopo g) noexcept
        : _g(std::move(g)) {}

    auto Modifier::take() noexcept -> GraphTopo {
        return std::move(_g);
    }

    auto Modifier::insert(Bridge bridge) noexcept -> BridgePos {
        auto n = bridge.node;
        auto idx = bridge.edge.index();
        auto &g = *_g._impl;
        auto passConnections = 0ul,
             passEdges = g._globalInputsCount;
        for (auto i : common::range0_(n)) {
            auto const &n_ = g._nodes[i];
            passConnections + n_._inputsCount;
            passEdges += n_._localEdgesCount + n_._outputsCount;
        }

        if (bridge.edge.isInput()) {
            auto e0 = g._connections[passConnections + idx];
            auto e1 = passEdges;
            std::for_each(g._connections.begin() + passConnections, g._connections.end(),
                          [=](auto &c) { if (c >= e1) { ++c; } });
            g._nodes.insert(g._nodes.begin() + n, 1, Node{0, 1, 1});
            g._connections.insert(g._connections.begin() + passConnections++, 1, e0);
            g._connections[passConnections + idx] = e1 - 1;
            return {n, e1};
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
            return {n, 0};
        }
    }

}// namespace refactor::graph_topo
