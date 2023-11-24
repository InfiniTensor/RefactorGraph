#include "computation/graph.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/simple_unary.h"
#include "computation/operators/transpose.h"
#include <gtest/gtest.h>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestMatMulTransposeGraphBuild1() {
        absl::InlinedVector<uint32_t, 4> perm = {0, 1, 3, 2};
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Transpose>(perm), "transpose0"};
        nodes[1] = Node{std::make_unique<Transpose>(perm), "transpose1"};
        nodes[2] = Node{std::make_unique<MatMul>(1.0, 1.0, false, false), "matmul"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 3, 5}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 5, 3}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {1, 3, 5, 3}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {2, 3, 3, 5}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {2, 3, 5, 5}, LayoutType::Others);

        return {
            {
                {0, {{0}, {2}}},
                {1, {{1}, {3}}},
                {2, {{2, 3}, {4}}},
            },
            {0, 1},// global inputs
            {4},   // global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input0"}},
                {1, {tensor1, "input1"}},
                {2, {tensor2, "input0_transpose"}},
                {3, {tensor3, "input1_transpose"}},
                {4, {tensor4, "output"}},
            },
        };
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestMatMulTransposeGraphBuild2() {
        absl::InlinedVector<uint32_t, 4> perm = {0, 1, 3, 2};
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<MatMul>(1.0, 1.0, false, false), "matmul"};
        nodes[1] = Node{std::make_unique<Transpose>(perm), "transpose1"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 3, 5}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 5, 4}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {2, 3, 3, 4}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {2, 3, 4, 3}, LayoutType::Others);

        return {
            {
                {0, {{0, 1}, {2}}},
                {1, {{2}, {3}}},
            },
            {0, 1},// global inputs
            {3},   // global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input0"}},
                {1, {tensor1, "input1"}},
                {2, {tensor2, "matmul_output"}},
                {3, {tensor3, "output"}},
            },
        };
    }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestMatMulTransposeGraphBuild3() {
        absl::InlinedVector<uint32_t, 4> perm = {0, 1, 3, 2};
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_unique<Transpose>(perm), "transpose0"};
        nodes[1] = Node{std::make_unique<Transpose>(perm), "transpose1"};
        nodes[2] = Node{std::make_unique<MatMul>(1.0, 1.0, false, false), "matmul"};
        nodes[3] = Node{std::make_unique<Transpose>(perm), "transpose3"};
        nodes[4] = Node{std::make_unique<SimpleUnary>(refactor::kernel::SimpleUnaryType::Relu), "relu"};


        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 3, 4}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 5, 3}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {1, 3, 4, 3}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {2, 3, 3, 5}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {2, 3, 4, 5}, LayoutType::Others);
        auto tensor5 = Tensor::share(DataType::F32, {2, 3, 5, 4}, LayoutType::Others);
        auto tensor6 = Tensor::share(DataType::F32, {2, 3, 5, 4}, LayoutType::Others);

        return {
            {
                {0, {{0}, {2}}},
                {1, {{1}, {3}}},
                {2, {{2, 3}, {4}}},
                {3, {{4}, {5}}},
                {4, {{5}, {6}}},
            },
            {0, 1},// global inputs
            {6},   // global outputs
            std::move(nodes),
            {
                {0, {tensor0, "input0"}},
                {1, {tensor1, "input1"}},
                {2, {tensor2, "input0_transpose"}},
                {3, {tensor3, "input1_transpose"}},
                {4, {tensor4, "matmul_output"}},
                {5, {tensor5, "transpose_output"}},
                {6, {tensor6, "output"}},
            },
        };
    }

    TEST(Graph, MatMulTranspose1) {
        auto graphTopo = TestMatMulTransposeGraphBuild1().build();
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

    TEST(Graph, MatMulTranspose2) {
        auto graphTopo = TestMatMulTransposeGraphBuild2().build();
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

    TEST(Graph, MatMulTranspose3) {
        auto graphTopo = TestMatMulTransposeGraphBuild3().build();
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
