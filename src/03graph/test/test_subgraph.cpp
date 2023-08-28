#include "graph/graph.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(Graph, Subgraph) {
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
    auto topo = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>();
    {
        auto a = topo.addEdge({Tensor{DataType::F32, {2, 3, 4, 5}}});
        auto b = topo.addEdge({Tensor{DataType::F32, {3, 1, 1}}});
        auto add = topo.addNode({NodeInfo{Operator{OpType::Add}}}, {a, b}, {EdgeInfo{}});
        auto output = add[0];
        topo.markOutput(output);
    }
    GraphMut graph(std::move(topo));
    {
        auto outputs = graph.topoMut().globalOutputs();
        graph.fillEdgeInfo();
        EXPECT_EQ(outputs.size(), 1);
        EXPECT_EQ(outputs[0].info().value, (EdgeInfo{Tensor{DataType::F32, {2, 3, 4, 5}}}));
    }
    {
        auto subgraphs = graph.extract({{graph.topoMut().nodes()[0]}});
        EXPECT_EQ(subgraphs.size(), 1);

        auto subgraph = subgraphs[0];
        EXPECT_EQ(subgraph.index(), 0);
        EXPECT_TRUE(subgraph.info().value.isSubgraph());
        EXPECT_TRUE(subgraph.info().value.subgraph().graph.get());

        auto subTopo = subgraph.info().value.subgraph().graph->topoMut();
        EXPECT_EQ(subTopo.nodes().size(), 1);
        EXPECT_EQ(subTopo.edges().size(), 3);
        EXPECT_EQ(subTopo.globalInputs().size(), 2);
        EXPECT_EQ(subTopo.globalOutputs().size(), 1);

        auto node = subTopo.nodes()[0].info().value;
        ASSERT_TRUE(node.isOperator());
        EXPECT_EQ(node.operator_().opType, OpType::Add);
        EXPECT_TRUE(node.operator_().attributes.empty());
    }
    graph.reduce();
    ASSERT_EQ(graph.topoMut().nodes().size(), 1);
    ASSERT_TRUE(graph.topoMut().nodes()[0].info().value.isOperator());
    {
        auto op = graph.topoMut().nodes()[0].info();
        EXPECT_EQ(op.value.operator_().opType, OpType::Add);
    }
}
