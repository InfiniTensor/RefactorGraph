#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/simple_unary.h"
#include <gtest/gtest.h>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestConvToMatMulGraphBuild1() {
        auto nodes = std::unordered_map<size_t, Node>{};
        int64_t dilations[2] = {1, 1};
        int64_t strides[2] = {1, 1};
        int64_t pads[4] = {0, 0, 0, 0};
        nodes[0] = Node{std::make_unique<Conv>(PoolAttributes(2, &dilations[0], &pads[0], &strides[0])), "conv"};
        nodes[1] = Node{std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "relu"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 1, 1}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {1, 2, 5, 5}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {1, 2, 5, 5}, LayoutType::Others);

        return {
            {
                {0, {{0, 1}, {2}}},
                {1, {{2}, {3}}},
            },
            {0, 1},// global inputs
            {3},   // global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "weight"}},
                {2, {tensor2, "conv_output"}},
                {3, {tensor3, "output"}},
            },
        };
    }

    TEST(Graph, ConvToMatMul1) {
        auto graphTopo = TestConvToMatMulGraphBuild1().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.optimize();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        fmt::println("Nodes info :");
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("\n Edges info :");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i, g_.edges[i].name,
                         vec2str(g_.edges[i].tensor->shape), g_.edges[i].tensor->layout.name());
        }
        ASSERT_EQ(g_.nodes.size(), 8);
        ASSERT_EQ(g_.edges.size(), 13);
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestConvToMatMulGraphBuild2() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv0"};
        nodes[1] = Node{std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "relu0"};
        nodes[2] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv1"};
        nodes[3] = Node{std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "relu1"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 1, 1}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {1, 2, 5, 5}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {1, 2, 5, 5}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {4, 3, 1, 1}, LayoutType::Others);
        auto tensor5 = Tensor::share(DataType::F32, {1, 4, 5, 5}, LayoutType::Others);
        auto tensor6 = Tensor::share(DataType::F32, {1, 4, 5, 5}, LayoutType::Others);

        return {
            {
                {0, {{0, 1}, {2}}},
                {1, {{2}, {3}}},
                {2, {{3, 4}, {5}}},
                {3, {{5}, {6}}},
            },
            {0, 1, 4},// global inputs
            {6},      // global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input0"}},
                {1, {tensor1, "weight0"}},
                {2, {tensor2, "conv0_output"}},
                {3, {tensor3, "relu0_output"}},
                {4, {tensor4, "weight1"}},
                {5, {tensor5, "conv1_output"}},
                {6, {tensor6, "output"}},
            },
        };
    }

    TEST(Graph, ConvToMatMul2) {
        auto graphTopo = TestConvToMatMulGraphBuild2().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.optimize();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        fmt::println("Nodes info :");
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("\n Edges info :");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i, g_.edges[i].name,
                         vec2str(g_.edges[i].tensor->shape), g_.edges[i].tensor->layout.name());
        }
        ASSERT_EQ(g_.nodes.size(), 16);
        ASSERT_EQ(g_.edges.size(), 25);
    }
}// namespace refactor::computation
