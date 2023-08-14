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
    auto a = topo.addEdge("a");
    auto b = topo.addEdge("b");
    auto add = topo.addNode("add", {a, b}, {"c"});
    auto c = add[0];
    auto d = topo.addEdge("d");
    auto mul = topo.addNode("mul", {c, d}, {"e"});
    auto e = mul[0];
    topo.markOutput({e});

    auto searcher = GraphTopoSearcher(std::move(topo));
    auto nodes = searcher.nodes();
    for (auto node : nodes) {
        std::cout << node.info() << std::endl;
    }
}
