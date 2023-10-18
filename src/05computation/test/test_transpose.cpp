#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/reshape.h"
#include "computation/operators/simple_unary.h"
#include "graph_topo/graph_topo.h"
#include "refactor/common.h"
#include <gtest/gtest.h>
#include <numeric>
#include <string.h>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild() {
        absl::InlinedVector<uint_lv2, 4> shape = {1, 3, 5, 5};
        absl::InlinedVector<uint_lv2, 4> shape1 = {2, 3, 3, 3};
        absl::InlinedVector<uint_lv2, 4> shape2 = {1, 2, 3, 3};
        absl::InlinedVector<uint_lv2, 4> shape3 = {4};
        absl::InlinedVector<uint_lv2, 4> shape4 = {2, 1, 3, 3};
        size_t num1 = 54;
        std::vector<float> weight(num1);
        std::iota(weight.begin(), weight.end(), 1.0);
        auto tensor0 = Tensor::share(DataType::F32, shape, LayoutType::NCHW, nullptr);   //0
        auto tensor1 = Tensor::share(DataType::F32, shape1, LayoutType::NCHW, nullptr);  //1
        auto tensor2 = Tensor::share(DataType::F32, shape2, LayoutType::NCHW, nullptr);  //2
        auto tensor3 = Tensor::share(DataType::I64, shape3, LayoutType::Others, nullptr);//3
        auto tensor4 = Tensor::share(DataType::F32, shape4, LayoutType::NCHW, nullptr);  //4
        auto ptr1 = reinterpret_cast<float *>(tensor1->malloc());
        memcpy(ptr1, weight.data(), num1 * sizeof(float));
        auto ptr2 = reinterpret_cast<int64_t *>(tensor3->malloc());
        ptr2[0] = 2;
        ptr2[1] = 1;
        ptr2[2] = 3;
        ptr2[3] = 3;
        Edge edge0 = {tensor0, "input"};
        Edge edge1 = {tensor1, "weight"};
        Edge edge2 = {tensor2, "conv_output"};
        Edge edge3 = {tensor3, "reshape_shape"};
        Edge edge4 = {tensor4, "output"};
        std::vector<int64_t> dilations = {1, 1};
        std::vector<int64_t> strides = {1, 1};
        std::vector<int64_t> paddings = {0, 0, 0, 0};
        PoolAttributes attr(2, dilations.data(), paddings.data(), strides.data());
        Node conv = {std::make_unique<Conv>(attr), "conv"};
        Node reshape = {std::make_unique<Reshape>(), "reshape"};
        std::unordered_map<size_t, refactor::graph_topo::BuilderNode<size_t>> topology = {
            {0, {{0, 1}, {2}}},
            {1, {{2, 3}, {4}}},
        };
        std::vector<size_t>
            globalInputs = {0, 1, 3};
        std::vector<size_t> globalOutputs = {4};
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes.emplace(0, std::move(conv));
        nodes.emplace(1, std::move(reshape));
        std::unordered_map<size_t, Edge> edges = {{0, {edge0}},
                                                  {1, {edge1}},
                                                  {2, {edge2}},
                                                  {3, {edge3}},
                                                  {4, {edge4}}};
        return {topology, globalInputs, globalOutputs, std::move(nodes), edges};
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild1() {
        absl::InlinedVector<uint_lv2, 4> shape0 = {1, 3, 5, 5};
        absl::InlinedVector<uint_lv2, 4> shape1 = {2, 3, 3, 3};
        absl::InlinedVector<uint_lv2, 4> shape2 = {2};
        absl::InlinedVector<uint_lv2, 4> shape3 = {1, 2, 3, 3};
        size_t num1 = 54;
        std::vector<float> weight(num1);
        std::iota(weight.begin(), weight.end(), 1.0);
        std::vector<float> bias = {1.0, 1.0};
        auto tensor0 = Tensor::share(DataType::F32, shape0, LayoutType::NCHW, nullptr);  //0
        auto tensor1 = Tensor::share(DataType::F32, shape1, LayoutType::NCHW, nullptr);  //1
        auto tensor2 = Tensor::share(DataType::F32, shape2, LayoutType::Others, nullptr);//2
        auto tensor3 = Tensor::share(DataType::I64, shape3, LayoutType::NCHW, nullptr);  //3
        auto tensor4 = Tensor::share(DataType::F32, shape3, LayoutType::NCHW, nullptr);  //4
        auto ptr1 = reinterpret_cast<float *>(tensor1->malloc());
        memcpy(ptr1, weight.data(), num1 * sizeof(float));
        auto ptr2 = reinterpret_cast<float *>(tensor2->malloc());
        memcpy(ptr2, bias.data(), 2 * sizeof(float));
        Edge edge0 = {tensor0, "input"};
        Edge edge1 = {tensor1, "weight"};
        Edge edge2 = {tensor2, "bias"};
        Edge edge3 = {tensor3, "conv_coutput"};
        Edge edge4 = {tensor4, "output"};
        std::vector<int64_t> dilations = {1, 1};
        std::vector<int64_t> strides = {1, 1};
        std::vector<int64_t> paddings = {0, 0, 0, 0};
        PoolAttributes attr(2, dilations.data(), paddings.data(), strides.data());
        Node conv = {std::make_unique<Conv>(attr), "conv"};
        Node unary = {std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "unary"};
        std::unordered_map<size_t, refactor::graph_topo::BuilderNode<size_t>> topology = {
            {0, {{0, 1, 2}, {3}}},
            {1, {{3}, {4}}},
        };
        std::vector<size_t>
            globalInputs = {0, 1, 2};
        std::vector<size_t> globalOutputs = {4};
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes.emplace(0, std::move(conv));
        nodes.emplace(1, std::move(unary));
        std::unordered_map<size_t, Edge> edges = {{0, {edge0}},
                                                  {1, {edge1}},
                                                  {2, {edge2}},
                                                  {3, {edge3}},
                                                  {4, {edge4}}};
        return {topology, globalInputs, globalOutputs, std::move(nodes), edges};
    }

    TEST(Graph, TransposeNHWC1) {
        auto graphTopo = TestGraphBuild().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.transpose();
        auto const &internal = std::move(g.internal());
        fmt::println("{}", internal.topology.toString());
        for (size_t i = 0; i < internal.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, internal.nodes[i].name);
        }
        fmt::println("============================================================");
        for (size_t i = 0; i < internal.edges.size(); ++i) {
            fmt::println("{}. \"{}\"", i, internal.edges[i].name);
        }
        ASSERT_EQ(internal.nodes.size(), 4);
        ASSERT_EQ(internal.edges.size(), 7);
        ASSERT_EQ(internal.nodes[0].name, "InsertTranspose0");
        ASSERT_EQ(internal.nodes[1].name, "conv");
        ASSERT_EQ(internal.nodes[2].name, "InsertTranspose1");
        ASSERT_EQ(internal.nodes[3].name, "reshape");
        ASSERT_EQ(internal.edges[0].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(internal.edges[1].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(internal.edges[2].tensor->layout, LayoutType::Others);
        ASSERT_EQ(internal.edges[3].tensor->layout, LayoutType::NHWC);
        //check weight converted
        auto tensor = internal.edges[1].tensor;
        float weight_t[]{1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0, 5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0,
                         16.0, 25.0, 8.0, 17.0, 26.0, 9.0, 18.0, 27.0, 28.0, 37.0, 46.0, 29.0, 38.0, 47.0, 30.0, 39.0, 48.0, 31.0, 40.0,
                         49.0, 32.0, 41.0, 50.0, 33.0, 42.0, 51.0, 34.0, 43.0, 52.0, 35.0, 44.0, 53.0, 36.0, 45.0, 54.0};
        ASSERT_TRUE(std::equal(weight_t, weight_t + tensor->elementsSize(), tensor->data->get<float>()));
    }

    TEST(Graph, TransposeNHWC2) {
        auto graphTopo = TestGraphBuild1().build();
        fmt::println("{}", graphTopo.topology.toString());
        Graph g(std::move(graphTopo));
        g.transpose();
        auto const &internal = std::move(g.internal());
        fmt::println("{}", internal.topology.toString());
        for (size_t i = 0; i < internal.nodes.size(); ++i) {
            fmt::println("{}. \"{}\"", i, internal.nodes[i].name);
        }
        fmt::println("============================================================");
        for (size_t i = 0; i < internal.edges.size(); ++i) {
            fmt::println("{}. \"{}\"", i, internal.edges[i].name);
        }
        ASSERT_EQ(internal.nodes.size(), 4);
        ASSERT_EQ(internal.edges.size(), 7);
        ASSERT_EQ(internal.nodes[0].name, "InsertTranspose0");
        ASSERT_EQ(internal.nodes[1].name, "conv");
        ASSERT_EQ(internal.nodes[2].name, "unary");
        ASSERT_EQ(internal.nodes[3].name, "InsertTranspose1");
        ASSERT_EQ(internal.edges[0].tensor->layout, LayoutType::NCHW);
        ASSERT_EQ(internal.edges[1].tensor->layout, LayoutType::NHWC);
        ASSERT_EQ(internal.edges[2].tensor->layout, LayoutType::Others);
        ASSERT_EQ(internal.edges[3].tensor->layout, LayoutType::NHWC);
        //check weight converted
        auto tensor = internal.edges[1].tensor;
        float weight_t[]{1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0, 5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0,
                         16.0, 25.0, 8.0, 17.0, 26.0, 9.0, 18.0, 27.0, 28.0, 37.0, 46.0, 29.0, 38.0, 47.0, 30.0, 39.0, 48.0, 31.0, 40.0,
                         49.0, 32.0, 41.0, 50.0, 33.0, 42.0, 51.0, 34.0, 43.0, 52.0, 35.0, 44.0, 53.0, 36.0, 45.0, 54.0};
        ASSERT_TRUE(std::equal(weight_t, weight_t + tensor->elementsSize(), tensor->data->get<float>()));
    }
}// namespace refactor::computation
