#include "error_handler.h"
#include "graph_topo.hpp"
#include "graph_topo_searcher.hpp"
#include <gtest/gtest.h>


TEST(GraphTopo, Build) {
    auto topo = GraphTopo<const char *, const char *>();
    auto a = topo.addEdge("a");
    auto b = topo.addEdge("b");
    auto add = topo.addNode("add", {a, b}, {"c"});
    auto c = add[0];
    auto d = topo.addEdge("d");
    auto mul = topo.addNode("mul", {c, d}, {"e"});
    auto e = mul[0];
    topo.markOutput({e});
}

TEST(GraphTopoSearcher, Build) {
    auto topo = GraphTopo<const char *, const char *>();
    auto a = topo.addEdge("a");                   // edge 0 | globalInput 0
    auto b = topo.addEdge("b");                   // edge 1 | globalInput 1
    auto add = topo.addNode("add", {a, b}, {"c"});// node 0 |
    auto c = add[0];                              // edge 2 |
    auto d = topo.addEdge("d");                   // edge 3 | globalInput 2
    auto mul = topo.addNode("mul", {c, d}, {"e"});// node 1 |
    auto e = mul[0];                              // edge 4 | globalOutput 0
    topo.markOutput({e});

    auto searcher = GraphTopoSearcher(std::move(topo));
    auto nodes = searcher.nodes();
    auto edges = searcher.edges();
    auto globalInputs = searcher.globalInputs();
    auto globalOutputs = searcher.globalOutputs();

    EXPECT_EQ(2, nodes.size());
    EXPECT_EQ(5, edges.size());
    EXPECT_EQ(3, globalInputs.size());
    EXPECT_EQ(1, globalOutputs.size());
    EXPECT_EQ("add", nodes[0].info());
    EXPECT_EQ("mul", nodes[1].info());
    EXPECT_EQ("a", edges[0].info());
    EXPECT_EQ("b", edges[1].info());
    EXPECT_EQ("c", edges[2].info());
    EXPECT_EQ("d", edges[3].info());
    EXPECT_EQ("e", edges[4].info());
    EXPECT_EQ("a", globalInputs[0].info());
    EXPECT_EQ("b", globalInputs[1].info());
    EXPECT_EQ("d", globalInputs[2].info());
    EXPECT_EQ("e", globalOutputs[0].info());
}
