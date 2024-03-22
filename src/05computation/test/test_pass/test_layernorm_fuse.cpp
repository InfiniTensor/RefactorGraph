#include "computation/graph.h"
#include "computation/operators/reduce.h"
#include "computation/operators/simple_binary.h"
#include "computation/operators/simple_unary.h"
#include <gtest/gtest.h>
#include <numeric>

namespace refactor::computation {
    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestLayernormFuseGraphBuild() {
        auto nodes = std::unordered_map<size_t, Node>{};
        absl::InlinedVector<uint32_t, 4> axes = {2};
        uint32_t rank = 3;
        bool keepDims = true;
        nodes[0] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Add), "add1"};
        nodes[1] = Node{std::make_unique<Reduce>(refactor::kernel::ReduceType::Mean, axes, rank, keepDims), "reducemean1"};
        nodes[2] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Sub), "sub"};
        nodes[3] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Pow), "pow"};
        nodes[4] = Node{std::make_unique<Reduce>(refactor::kernel::ReduceType::Mean, axes, rank, keepDims), "reducemean1"};
        nodes[5] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Add), "add2"};
        nodes[6] = Node{std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Sqrt), "sqrt"};
        nodes[7] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Div), "div"};
        nodes[8] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Mul), "mul"};
        nodes[9] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Add), "add3"};
        nodes[10] = Node{std::make_unique<SimpleBinary>(refactor::kernel::SimpleBinaryType::Add), "add4"};


        auto tensor0 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {1, 101, 768}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {64, 101, 1}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor5 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor6 = Tensor::share(DataType::F32, {64, 101, 1}, LayoutType::Others);
        auto tensor7 = Tensor::share(DataType::F32, {}, LayoutType::Others);
        auto tensor8 = Tensor::share(DataType::F32, {64, 101, 1}, LayoutType::Others);
        auto tensor9 = Tensor::share(DataType::F32, {64, 101, 1}, LayoutType::Others);
        auto tensor10 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor11 = Tensor::share(DataType::F32, {768}, LayoutType::Others);
        auto tensor12 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor13 = Tensor::share(DataType::F32, {768}, LayoutType::Others);
        auto tensor14 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);
        auto tensor15 = Tensor::share(DataType::F32, {}, LayoutType::Others);
        auto tensor16 = Tensor::share(DataType::F32, {64, 101, 768}, LayoutType::Others);

        auto scale = reinterpret_cast<float *>(tensor11->malloc());
        std::iota(scale, scale + tensor11->elementsSize(), 1.0);
        auto bias = reinterpret_cast<float *>(tensor13->malloc());
        std::iota(bias, bias + tensor13->elementsSize(), 0.0);
        float epsilon_ = 0.000009999999747378752;
        std::memcpy(tensor7->malloc(), &epsilon_, tensor7->bytesSize());
        float pow = 2.0;
        std::memcpy(tensor15->malloc(), &pow, tensor15->bytesSize());

        return {
            {
                {0, {{0, 1}, {2}}},   //add
                {1, {{2}, {3}}},      //reducemean
                {2, {{3, 2}, {4}}},   //sub
                {3, {{4, 15}, {5}}},  //pow
                {4, {{5}, {6}}},      //reducemean
                {5, {{6, 7}, {8}}},   //add
                {6, {{8}, {9}}},      //sqrt
                {7, {{9, 4}, {10}}},  //div
                {8, {{10, 11}, {12}}},//mul
                {9, {{12, 13}, {14}}},//add
                {10, {{14, 2}, {16}}},
            },
            {
                0,
                1,
                7,
                11,
                13,
                15,
            },   // global inputs
            {16},// global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input0"}},
                {1, {tensor1, "input1"}},
                {2, {tensor2, "add1_output"}},
                {3, {tensor3, "reducemean1_output"}},
                {4, {tensor4, "sub_output"}},
                {5, {tensor5, "pow_output"}},
                {6, {tensor6, "reducemean2_output"}},
                {7, {tensor7, "add2_input"}},
                {8, {tensor8, "add2_output"}},
                {9, {tensor9, "sqrt_output"}},
                {10, {tensor10, "div_output"}},
                {11, {tensor11, "mul_input"}},
                {12, {tensor12, "mul_output"}},
                {13, {tensor13, "add3_input"}},
                {14, {tensor14, "output"}},
                {15, {tensor15, "pow_input"}},
                {16, {tensor15, "add4_output"}},
            },
        };
    }

    TEST(Graph, LayerNormFuse) {
        auto graphTopo = TestLayernormFuseGraphBuild().build();
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
    }
}// namespace refactor::computation