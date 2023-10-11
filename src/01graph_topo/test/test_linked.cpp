#include "graph_topo/linked_graph.h"
#include "topo.h"
#include <gtest/gtest.h>

using namespace refactor::graph_topo;

TEST(GraphTopo, LinkedGraph) {
    auto g = testTopo().build();
    LinkedGraph<const char *, const char *> g_(g);
    // auto g__ = g_.build();
    // EXPECT_EQ(g__.nodes, g.nodes);
    // EXPECT_EQ(g__.edges, g.edges);
}
