#include "graph/graph.h"
#include "transformation/conv.h"
#include <gtest/gtest.h>


namespace refactor::transformation {

    using GraphTopoCell =
        GraphTopo<refactor::graph::Cell<refactor::graph::NodeInfo>,
                  refactor::graph::Cell<refactor::graph::EdgeInfo>>;
    using NodeInfo = refactor::graph::NodeInfo;
    using EdgeInfo = refactor::graph::EdgeInfo;
    using Tensor = refactor::graph::Tensor;
    using ShapeVariable = refactor::graph::ShapeVariable;
    using GraphMut = refactor::graph::GraphMut;
    using Operator = refactor::graph::Operator;
    using DataType = refactor::common::DataType;
    using OpType = refactor::common::OpType;

    TEST(Transformation, Conv1x1) {
        auto topo = GraphTopoCell();
        // create origin graph
        {
            auto a = topo.addEdge({Tensor{DataType::F32, {1, 3, 224, 224}}});
            auto b = topo.addEdge({Tensor{DataType::F32, {2, 3, 1, 1}}});
            auto conv = topo.addNode({NodeInfo{Operator{OpType::Conv}}}, {a, b}, {EdgeInfo{}});
            auto output = conv[0];
            topo.markOutput(output);
        }
        GraphMut graph(std::move(topo));
        graph.fillEdgeInfo();
        auto output = graph.topoMut().globalOutputs();
        EXPECT_EQ(output.size(), 1);
        EXPECT_EQ(output[0].info().value, (EdgeInfo{Tensor{DataType::F32, {1, 2, 224, 224}}}));
        // create ConvTransformer
        auto convTransformer = ConvTransformer(std::move(graph));
        EXPECT_EQ(convTransformer._convs.size(), 1);
        EXPECT_EQ(convTransformer._convs[convTransformer._graph.topoMut().nodes()[0]], convTransformer.Rule::_1x1);
        EXPECT_TRUE(convTransformer._graph.topoMut().nodes()[0].info().value.isSubgraph());
        convTransformer.transform();
        auto subTopo = convTransformer._graph.topoMut().nodes()[0].info().value.subgraph().graph->topoMut();
        {
            EXPECT_EQ(subTopo.nodes().size(), 7);
            EXPECT_EQ(subTopo.edges().size(), 12);
            EXPECT_EQ(subTopo.globalInputs().size(), 5);
            EXPECT_EQ(subTopo.globalOutputs().size(), 1);
            EXPECT_EQ(subTopo.nodes()[0].info().value.operator_().opType, OpType::Transpose);
            EXPECT_EQ(subTopo.nodes()[1].info().value.operator_().opType, OpType::Transpose);
            EXPECT_EQ(subTopo.nodes()[2].info().value.operator_().opType, OpType::Reshape);
            EXPECT_EQ(subTopo.nodes()[3].info().value.operator_().opType, OpType::Reshape);
            EXPECT_EQ(subTopo.nodes()[4].info().value.operator_().opType, OpType::Gemm);
            EXPECT_EQ(subTopo.nodes()[5].info().value.operator_().opType, OpType::Reshape);
            EXPECT_EQ(subTopo.nodes()[6].info().value.operator_().opType, OpType::Transpose);
        }
        EXPECT_EQ(subTopo.globalOutputs()[0].info().value, (EdgeInfo{Tensor{DataType::F32, {1, 2, 224, 224}}}));
    }

}// namespace refactor::transformation
