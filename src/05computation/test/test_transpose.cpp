#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/reshape.h"
#include "graph_topo/graph_topo.h"
#include <gtest/gtest.h>
#include <numeric>
#include <string.h>

namespace refactor::computation {
    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestGraphBuild() {
        refactor::graph_topo::Builder<size_t, Node, size_t, Edge> builder;
        absl::InlinedVector<int64_t, 4> shape = {1, 3, 5, 5};
        absl::InlinedVector<int64_t, 4> shape1 = {2, 3, 3, 3};
        absl::InlinedVector<int64_t, 4> shape2 = {1, 2, 3, 3};
        absl::InlinedVector<int64_t, 4> shape3 = {4};
        absl::InlinedVector<int64_t, 4> shape4 = {2, 1, 3, 3};
        size_t num1 = 54;
        std::vector<float> weight(num1);
        std::iota(weight.begin(), weight.end(), 1.0);
        auto tensor0 = Tensor::share(refactor::common::DataType::F32, shape, LayoutType::NCHW, nullptr);   //0
        auto tensor1 = Tensor::share(refactor::common::DataType::F32, shape1, LayoutType::NCHW, nullptr);  //1
        auto tensor2 = Tensor::share(refactor::common::DataType::F32, shape2, LayoutType::NCHW, nullptr);  //2
        auto tensor3 = Tensor::share(refactor::common::DataType::I64, shape3, LayoutType::Others, nullptr);//3                                                                                    //3
        auto tensor4 = Tensor::share(refactor::common::DataType::F32, shape4, LayoutType::NCHW, nullptr);  //4
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
        Node conv = {std::make_shared<Conv>(), "conv"};         //0
        Node reshape = {std::make_shared<Reshape>(), "reshape"};//1
        std::unordered_map<size_t, refactor::graph_topo::BuilderNode<size_t>> topology = {
            {0, {{0, 1}, {2}}},
            {1, {{2, 3}, {4}}},
        };
        std::vector<size_t>
            globalInputs = {0, 1, 3};
        std::vector<size_t> globalOutputs = {4};
        std::unordered_map<size_t, Node> nodes = {{0, {conv}},
                                                  {1, {reshape}}};
        std::unordered_map<size_t, Edge> edges = {{0, {edge0}},
                                                  {1, {edge1}},
                                                  {2, {edge2}},
                                                  {3, {edge3}},
                                                  {4, {edge4}}};
        builder = {topology, globalInputs, globalOutputs, nodes, edges};
        return builder;
    }

    TEST(Graph, TransposeNHWC) {
        auto graphTopo = TestGraphBuild().build();
        Graph g(graphTopo);
        g.transpose();
        auto const _internal = g.internal();
        ASSERT_EQ(_internal.nodes.size(), 4);
        ASSERT_EQ(_internal.edges.size(), 7);
    }
}// namespace refactor::computation
