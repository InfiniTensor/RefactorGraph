#include "graph/graph.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(Graph, GraphMut) {
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
    auto topo = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>();
    auto input = topo.addEdge({Tensor{DataType::F32, {1, 3, 224, 224}}});
    auto kernel = topo.addEdge({Tensor{DataType::F32, {6, 3, 3, 3}}});
    auto conv = topo.addNode({NodeInfo{Operator{OpType::Conv}}}, {input, kernel}, {EdgeInfo{}});
    auto output = conv[0];
    topo.markOutput(output);

    auto graphMut = GraphMut(std::move(topo));
    graphMut.fillEdgeInfo();
    auto otherTopo = graphMut.intoGraphTopo();

    auto graph = Graph(std::move(otherTopo));
    auto searcher = graph.topo();

    auto nodes = searcher.nodes();
    EXPECT_EQ(nodes.size(), 1);
    EXPECT_EQ(nodes[0].info(), NodeInfo{Operator{OpType::Conv}});

    auto edges = searcher.edges();
    EXPECT_EQ(edges.size(), 3);
    EXPECT_EQ(edges[0].info(), (Tensor{DataType::F32, {1, 3, 224, 224}}));
    EXPECT_EQ(edges[1].info(), (Tensor{DataType::F32, {6, 3, 3, 3}}));
    EXPECT_EQ(edges[2].info(), (Tensor{DataType::F32, {1, 6, 222, 222}}));
}
