#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/reshape.h"
#include "computation/operators/simple_unary.h"
#include <gtest/gtest.h>
#include <numeric>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv0"};
        nodes[1] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv1"};
        nodes[2] = Node{std::make_unique<MatMul>(1.0, 1.0, false, false), "matmul"};
        nodes[3] = Node{std::make_unique<Reshape>(), "reshape0"};
        nodes[4] = Node{std::make_unique<Reshape>(), "reshape1"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::NCHW);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor2 = Tensor::share(DataType::F32, {2}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor4 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor5 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor6 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor7 = Tensor::share(DataType::I64, {4}, LayoutType::Others);
        auto tensor8 = Tensor::share(DataType::I64, {4}, LayoutType::Others);
        auto tensor9 = Tensor::share(DataType::F32, {2, 1, 3, 3}, LayoutType::NCHW);
        auto tensor10 = Tensor::share(DataType::F32, {2, 1, 3, 3}, LayoutType::NCHW);
        auto weight = reinterpret_cast<float *>(tensor1->malloc());
        std::iota(weight, weight + tensor1->elementsSize(), 1.0);
        int64_t dataShape[]{2, 1, 3, 3};
        std::memcpy(tensor7->malloc(), dataShape, tensor7->bytesSize());
        std::memcpy(tensor8->malloc(), dataShape, tensor8->bytesSize());

        return {
            {
                {0, {{0, 1, 2}, {5}}},
                {1, {{0, 3}, {4}}},
                {2, {{1, 3}, {6}}},
                {3, {{5, 7}, {9}}},
                {4, {{5, 8}, {10}}},
            },
            {0, 2, 3},    // global inputs
            {4, 6, 9, 10},// global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "weight0"}},
                {2, {tensor2, "bias"}},
                {3, {tensor3, "weight1"}},
                {4, {tensor4, "output1"}},
                {5, {tensor5, "conv_output"}},
                {6, {tensor6, "output2"}},
                {7, {tensor7, "shape1"}},
                {8, {tensor8, "shape2"}},
                {9, {tensor9, "output3"}},
                {10, {tensor10, "output4"}},
            },
        };
    }

    TEST(Graph, TransposeNHWC) {
        auto graphTopo = TestGraphBuild().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.layoutPermute();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        fmt::println("Nodes info :");
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("\n Edges info :");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i, g_.edges[i].name,
                         vecToString(g_.edges[i].tensor->shape), g_.edges[i].tensor->layout.name());
        }
        ASSERT_EQ(g_.nodes.size(), 9);
        ASSERT_EQ(g_.edges.size(), 16);
        ASSERT_EQ(g_.edges[0].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[1].tensor->layout, LayoutType::Others);
        ASSERT_EQ(g_.edges[2].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[3].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[6].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[7].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[8].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[9].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[10].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[11].tensor->layout, LayoutType::NCHW);
        //check weight converted
        auto tensor = g_.edges[8].tensor;
        float weight_t[]{1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0, 5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0,
                         16.0, 25.0, 8.0, 17.0, 26.0, 9.0, 18.0, 27.0, 28.0, 37.0, 46.0, 29.0, 38.0, 47.0, 30.0, 39.0, 48.0, 31.0, 40.0,
                         49.0, 32.0, 41.0, 50.0, 33.0, 42.0, 51.0, 34.0, 43.0, 52.0, 35.0, 44.0, 53.0, 36.0, 45.0, 54.0};
        ASSERT_TRUE(std::equal(weight_t, weight_t + tensor->elementsSize(), tensor->data->get<float>()));
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild1() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Reshape>(), "reshape"};
        nodes[1] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv"};
        nodes[2] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv1"};
        nodes[3] = Node{std::make_unique<Reshape>(), "reshape1"};

        auto tensor0 = Tensor::share(DataType::F32, {3, 1, 5, 5}, LayoutType::NCHW);
        auto tensor1 = Tensor::share(DataType::I64, {4}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::NCHW);
        auto tensor3 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor4 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor5 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor6 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor7 = Tensor::share(DataType::I64, {3}, LayoutType::Others);
        auto tensor8 = Tensor::share(DataType::F32, {3, 5, 5}, LayoutType::Others);

        int64_t data[]{1, 3, 5, 5};
        std::memcpy(tensor1->malloc(), data, tensor1->bytesSize());
        int64_t data1[]{3, 5, 5};
        std::memcpy(tensor7->malloc(), data1, tensor7->bytesSize());
        return {
            {
                {0, {{0, 1}, {2}}},
                {1, {{2, 3}, {5}}},
                {2, {{2, 4}, {6}}},
                {3, {{2, 7}, {8}}},
            },
            {0, 3, 4},// globalInputs
            {5, 6, 8},// globalOutputs
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "shape"}},
                {2, {tensor2, "reshape_output"}},
                {3, {tensor3, "weight1"}},
                {4, {tensor4, "weight2"}},
                {5, {tensor5, "output1"}},
                {6, {tensor6, "output2"}},
                {7, {tensor7, "shape1"}},
                {8, {tensor8, "output3"}},
            },
        };
    }

    TEST(Graph, TransposeNHWC1) {
        auto graphTopo = TestGraphBuild1().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.layoutPermute();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        fmt::println("Nodes info :");
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("\n Edges info :");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i, g_.edges[i].name,
                         vecToString(g_.edges[i].tensor->shape), g_.edges[i].tensor->layout.name());
        }
        ASSERT_EQ(g_.nodes.size(), 9);
        ASSERT_EQ(g_.edges.size(), 14);
    }

}// namespace refactor::computation
