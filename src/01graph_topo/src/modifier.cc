#include "graph_topo/modifier.h"
#include "common/error_handler.h"

namespace refactor::graph_topo {

    Modifier::Modifier(GraphTopo g) noexcept
        : _g(std::move(g)) {}

    GraphTopo Modifier::take() noexcept {
        return std::move(_g);
    }

    auto Modifier::insert(std::vector<Bridge>) noexcept -> std::vector<BridgePos> {
        TODO("");
    }

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

}// namespace refactor::graph_topo
