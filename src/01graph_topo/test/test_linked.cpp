#include "graph_topo/linked_graph.hpp"
#include "topo.h"
#include <gtest/gtest.h>

using namespace refactor::graph_topo;

TEST(GraphTopo, LinkedGraph) {
    auto g = testTopo().build();
    LinkedGraph<const char *, const char *> g_(g);
    fmt::println("{}", g_.toString());
    auto g__ = LinkedGraph(g_).intoGraph();
    EXPECT_EQ(g__.nodes, g.nodes);
    EXPECT_EQ(g__.edges, g.edges);

    auto n0 = g_.nodes()[0];
    auto n1 = g_.nodes()[1];
    auto n2 = g_.nodes()[2];

    auto n3 = g_.pushNode("n3", {g_.shareEdge("e0"), g_.shareEdge("e1")});
    n3->connect(0, n2->outputs()[0]);
    g_.setOutputs({n3->outputs()[1]});
    fmt::println("{}", g_.toString());

    n2->disconnect(0);
    fmt::println("{}", g_.toString());

    n2->connect(0, g_.outputs()[0]);
    fmt::println("{}", g_.toString());

    g_.eraseNode(n1);
    fmt::println("{}", g_.toString());

    EXPECT_FALSE(g_.sort());
    fmt::println("{}", g_.toString());

    n3->connect(0, n0->outputs()[1]);
    g_.cleanup();
    fmt::println("{}", g_.toString());
}
