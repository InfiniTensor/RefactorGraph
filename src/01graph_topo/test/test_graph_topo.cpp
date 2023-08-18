#include "graph_topo/graph_topo_searcher.hpp"
#include <gtest/gtest.h>

TEST(GraphTopo, Build) {
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
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
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
    auto topo = GraphTopo<const char *, const char *>();
    {
        auto a = topo.addEdge("a");                   // edge 0 | globalInput 0
        auto b = topo.addEdge("b");                   // edge 1 | globalInput 1
        auto add = topo.addNode("add", {a, b}, {"c"});// node 0 |
        auto c = add[0];                              // edge 2 |
        auto d = topo.addEdge("d");                   // edge 3 | globalInput 2
        auto mul = topo.addNode("mul", {c, d}, {"e"});// node 1 |
        auto e = mul[0];                              // edge 4 | globalOutput 0
        topo.markOutput({e});
    }
    auto searcher = GraphTopoSearcher(std::move(topo));
    {
        auto globalInputs = searcher.globalInputs();
        auto globalOutputs = searcher.globalOutputs();

        EXPECT_EQ(3, globalInputs.size());
        EXPECT_EQ("a", globalInputs[0].info());
        EXPECT_EQ("b", globalInputs[1].info());
        EXPECT_EQ("d", globalInputs[2].info());

        EXPECT_EQ(1, globalOutputs.size());
        EXPECT_EQ("e", globalOutputs[0].info());
    }
    auto nodes = searcher.nodes();
    auto edges = searcher.edges();
    {
        EXPECT_EQ(2, nodes.size());

        auto add = nodes[0];
        auto mul = nodes[1];

        EXPECT_EQ("add", add.info());
        EXPECT_EQ("mul", mul.info());

        EXPECT_EQ(2, add.inputs().size());
        EXPECT_EQ("a", add.inputs()[0].info());
        EXPECT_EQ("b", add.inputs()[1].info());
        EXPECT_EQ(1, add.outputs().size());
        EXPECT_EQ("c", add.outputs()[0].info());

        EXPECT_EQ(2, mul.inputs().size());
        EXPECT_EQ("c", mul.inputs()[0].info());
        EXPECT_EQ("d", mul.inputs()[1].info());
        EXPECT_EQ(1, mul.outputs().size());
        EXPECT_EQ("e", mul.outputs()[0].info());
    }
    {
        EXPECT_EQ(5, edges.size());

        auto a = edges[0];
        auto b = edges[1];
        auto c = edges[2];
        auto d = edges[3];
        auto e = edges[4];

        EXPECT_EQ("a", a.info());
        EXPECT_EQ("b", b.info());
        EXPECT_EQ("c", c.info());
        EXPECT_EQ("d", d.info());
        EXPECT_EQ("e", e.info());

        EXPECT_FALSE(a.source().exist());
        EXPECT_EQ(1, a.targets().size());
        EXPECT_EQ("add", a.targets()[0].info());

        EXPECT_FALSE(b.source().exist());
        EXPECT_EQ(1, b.targets().size());
        EXPECT_EQ("add", b.targets()[0].info());

        EXPECT_EQ("add", c.source().info());
        EXPECT_EQ(1, c.targets().size());
        EXPECT_EQ("mul", c.targets()[0].info());

        EXPECT_FALSE(d.source().exist());
        EXPECT_EQ(1, d.targets().size());
        EXPECT_EQ("mul", d.targets()[0].info());

        EXPECT_EQ("mul", e.source().info());
        EXPECT_TRUE(e.targets().empty());
    }
    {
        std::set<const char *> nodeNames{"add", "mul"};
        for (auto node : nodes) {
            EXPECT_EQ(1, nodeNames.erase(node.info()));
        }
        EXPECT_TRUE(nodeNames.empty());
    }
    {
        std::set<const char *> edgeNames{"a", "b", "c", "d", "e"};
        for (auto edge : edges) {
            EXPECT_EQ(1, edgeNames.erase(edge.info()));
        }
        EXPECT_TRUE(edgeNames.empty());
    }
}

template<class T>
struct Cell {
    mutable T value;
    Cell(T &&value) : value(std::forward<T>(value)) {}
};

TEST(GraphTopo, Cell) {
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
    auto topo = GraphTopo<Cell<std::string>, Cell<std::string>>();
    {
        auto a = topo.addEdge({"a"});
        auto b = topo.addEdge({"b"});
        auto add = topo.addNode({"add"}, {a, b}, {{"c"}});
        auto c = add[0];
        auto d = topo.addEdge({"d"});
        auto mul = topo.addNode({"mul"}, {c, d}, {{"e"}});
        auto e = mul[0];
        topo.markOutput({e});
    }
    auto const searcher = GraphTopoSearcher(std::move(topo));
    for (auto const node : searcher.nodes()) {
        node.info().value += "!";
    }
    for (auto const edge : searcher.edges()) {
        edge.info().value += "?";
    }
    auto const nodes = searcher.nodes();
    EXPECT_EQ("add!", nodes[0].info().value);
    EXPECT_EQ("mul!", nodes[1].info().value);

    auto const edges = searcher.edges();
    EXPECT_EQ("a?", edges[0].info().value);
    EXPECT_EQ("b?", edges[1].info().value);
    EXPECT_EQ("c?", edges[2].info().value);
    EXPECT_EQ("d?", edges[3].info().value);
    EXPECT_EQ("e?", edges[4].info().value);
}
