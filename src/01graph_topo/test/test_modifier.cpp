#include "topo.h"
#include <gtest/gtest.h>

using namespace refactor::graph_topo;

TEST(GraphTopo, InplaceModifier) {
    auto [topology, nodes, edges] = testTopo().build();
    fmt::println("{}", topology.toString());
    {
        auto modifier = InplaceModifier(topology);
        modifier.insert(Bridge{0, OnNode::input(1)});
        fmt::println("{}", modifier.take().toString());
    }
    {
        auto modifier = InplaceModifier(topology);
        modifier.insert(Bridge{1, OnNode::input(0)});
        fmt::println("{}", modifier.take().toString());
    }
    {
        auto modifier = InplaceModifier(topology);
        modifier.insert(Bridge{0, OnNode::output(1)});
        fmt::println("{}", modifier.take().toString());
    }
    {
        auto modifier = InplaceModifier(topology);
        modifier.insert(Bridge{2, OnNode::output(0)});
        fmt::println("{}", modifier.take().toString());
    }
}
