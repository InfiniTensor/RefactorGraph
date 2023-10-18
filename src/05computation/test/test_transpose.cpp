#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/reshape.h"
#include "computation/operators/simple_unary.h"
#include <gtest/gtest.h>
#include <numeric>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv"};
        nodes[1] = Node{std::make_unique<Reshape>(), "reshape"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::NCHW);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor2 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor3 = Tensor::share(DataType::I64, {4}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {2, 1, 3, 3}, LayoutType::NCHW);

        auto weight = reinterpret_cast<float *>(tensor1->malloc());
        std::iota(weight, weight + tensor1->elementsSize(), 1.0);
        int64_t data3[]{2, 1, 3, 3};
        std::memcpy(tensor3->malloc(), data3, tensor3->bytesSize());

        return {
            {
                {0, {{0, 1}, {2}}},
                {1, {{2, 3}, {4}}},
            },
            {0, 1, 3},// globalInputs
            {4},      // globalOutputs
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "weight"}},
                {2, {tensor2, "conv_output"}},
                {3, {tensor3, "reshape_shape"}},
                {4, {tensor4, "output"}},
            },
        };
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild1() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = {std::make_unique<Conv>(PoolAttributes(2, nullptr, nullptr, nullptr)), "conv"};
        nodes[1] = {std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "unary"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 5, 5}, LayoutType::NCHW);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 3, 3}, LayoutType::NCHW);
        auto tensor2 = Tensor::share(DataType::F32, {2}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::I64, {1, 2, 3, 3}, LayoutType::NCHW);
        auto tensor4 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);

        auto weight = reinterpret_cast<float *>(tensor1->malloc());
        std::iota(weight, weight + tensor1->elementsSize(), 1.0);
        float bias[]{1, 1};
        std::memcpy(tensor2->malloc(), bias, tensor2->bytesSize());

        return {
            {
                {0, {{0, 1, 2}, {3}}},
                {1, {{3}, {4}}},
            },
            {0, 1, 2},// globalInputs
            {4},      // globalOutputs
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "weight"}},
                {2, {tensor2, "bias"}},
                {3, {tensor3, "conv_coutput"}},
                {4, {tensor4, "output"}},
            },
        };
    }

    TEST(Graph, TransposeNHWC1) {
        auto graphTopo = TestGraphBuild().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.layoutPermute();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("============================================================");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.edges[i].name);
        }
        ASSERT_EQ(g_.nodes.size(), 4);
        ASSERT_EQ(g_.edges.size(), 7);
        ASSERT_EQ(g_.nodes[0].name, "InsertTranspose0");
        ASSERT_EQ(g_.nodes[1].name, "conv");
        ASSERT_EQ(g_.nodes[2].name, "InsertTranspose1");
        ASSERT_EQ(g_.nodes[3].name, "reshape");
        ASSERT_EQ(g_.edges[0].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[1].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[2].tensor->layout, LayoutType::Others);
        ASSERT_EQ(g_.edges[3].tensor->layout, LayoutType::NHWC);
        //check weight converted
        auto tensor = g_.edges[1].tensor;
        float weight_t[]{1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0, 5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0,
                         16.0, 25.0, 8.0, 17.0, 26.0, 9.0, 18.0, 27.0, 28.0, 37.0, 46.0, 29.0, 38.0, 47.0, 30.0, 39.0, 48.0, 31.0, 40.0,
                         49.0, 32.0, 41.0, 50.0, 33.0, 42.0, 51.0, 34.0, 43.0, 52.0, 35.0, 44.0, 53.0, 36.0, 45.0, 54.0};
        ASSERT_TRUE(std::equal(weight_t, weight_t + tensor->elementsSize(), tensor->data->get<float>()));
    }

    TEST(Graph, TransposeNHWC2) {
        auto graphTopo = TestGraphBuild1().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.layoutPermute();
        auto const &g_ = g.internal().contiguous();
        fmt::println("{}", g_.topology.toString());
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
        }
        fmt::println("============================================================");
        for (size_t i = 0; i < g_.edges.size(); ++i) {
            fmt::println("{}. \"{}\"", i, g_.edges[i].name);
        }
        ASSERT_EQ(g_.nodes.size(), 4);
        ASSERT_EQ(g_.edges.size(), 7);
        ASSERT_EQ(g_.nodes[0].name, "InsertTranspose0");
        ASSERT_EQ(g_.nodes[1].name, "conv");
        ASSERT_EQ(g_.nodes[2].name, "unary");
        ASSERT_EQ(g_.nodes[3].name, "InsertTranspose1");
        ASSERT_EQ(g_.edges[0].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(g_.edges[1].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(g_.edges[2].tensor->layout, LayoutType::Others);
        ASSERT_EQ(g_.edges[3].tensor->layout, LayoutType::NHWC);
        //check weight converted
        auto tensor = g_.edges[1].tensor;
        float weight_t[]{1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0, 5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0,
                         16.0, 25.0, 8.0, 17.0, 26.0, 9.0, 18.0, 27.0, 28.0, 37.0, 46.0, 29.0, 38.0, 47.0, 30.0, 39.0, 48.0, 31.0, 40.0,
                         49.0, 32.0, 41.0, 50.0, 33.0, 42.0, 51.0, 34.0, 43.0, 52.0, 35.0, 44.0, 53.0, 36.0, 45.0, 54.0};
        ASSERT_TRUE(std::equal(weight_t, weight_t + tensor->elementsSize(), tensor->data->get<float>()));
    }

}// namespace refactor::computation
