#include "topo.h"
#include <gtest/gtest.h>

using namespace refactor::graph_topo;

TEST(GraphTopo, Modifier) {
    auto [topology, nodes, edges] = testTopo().build();
    fmt::println("{}", topology.toString());

    auto modifier = Modifier(std::move(topology));
    modifier.insert(Bridge{0, OnNode::input(1)});
    topology = modifier.take();
    fmt::println("{}", topology.toString());
}
