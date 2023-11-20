#include "computation/graph_mutant.h"
#include "computation/mutant_generator.h"
#include "computation/operators/concat.h"
#include "computation/operators/conv.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/reshape.h"
#include "computation/operators/transpose.h"
#include <gtest/gtest.h>
#include <numeric>

namespace refactor::computation {

    // refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestInGraphBuild() {
    //     auto nodes = std::unordered_map<size_t, Node>{};
    //     nodes[0] = Node{std::make_shared<MatMulBox>(), "matmul_1"};
    //     nodes[1] = Node{std::make_shared<MatMulBox>(), "matmul_2"};
    //     nodes[2] = Node{std::make_shared<ConcatBox>(), "concat"};

    //     auto tensor0 = Tensor::share(DataType::F32, {5, 6}, LayoutType::Others);
    //     auto tensor1 = Tensor::share(DataType::F32, {4, 5}, LayoutType::Others);
    //     auto tensor2 = Tensor::share(DataType::F32, {5, 7}, LayoutType::Others);
    //     auto tensor3 = Tensor::share(DataType::F32, {4, 6}, LayoutType::Others);
    //     auto tensor4 = Tensor::share(DataType::F32, {4, 7}, LayoutType::Others);
    //     auto tensor5 = Tensor::share(DataType::F32, {4, 13}, LayoutType::Others);
    //     // initialize inputs data
    //     auto data0 = reinterpret_cast<float *>(tensor0->malloc());
    //     auto data1 = reinterpret_cast<float *>(tensor1->malloc());
    //     auto data2 = reinterpret_cast<float *>(tensor2->malloc());
    //     std::iota(data0, data0 + tensor0->elementsSize(), 1.0);
    //     std::iota(data1, data1 + tensor1->elementsSize(), 1.0);
    //     std::iota(data2, data2 + tensor2->elementsSize(), 1.0);
    //     // initialize outputs data
    //     float outputData[]{255.0, 270.0, 285.0, 300.0, 315.0, 330.0, 295.0, 310.0, 325.0, 340.0, 355.0, 370.0, 385.0, 580.0, 620.0, 660.0, 700.0,
    //                        740.0, 780.0, 670.0, 710.0, 750.0, 790.0, 830.0, 870.0, 910.0, 905.0, 970.0, 1035.0, 1100.0, 1165.0, 1230.0, 1045.0, 1110.0, 1175.0, 1240.0,
    //                        1305.0, 1370.0, 1435.0, 1230.0, 1320.0, 1410.0, 1500.0, 1590.0, 1680.0, 1420.0, 1510.0, 1600.0, 1690.0, 1780.0, 1870.0, 1960.};
    //     std::memcpy(tensor5->malloc(), outputData, tensor5->bytesSize());

    //     return {
    //         {{0, {{1, 0}, {3}}},
    //          {1, {{1, 2}, {4}}},
    //          {2, {{3, 4}, {5}}}},
    //         {0, 1, 2},
    //         {5},
    //         std::move(nodes),
    //         {
    //             {0, {tensor0, "input_tensor_0"}},
    //             {1, {tensor1, "input_tensor_1"}},
    //             {2, {tensor2, "input_tensor_2"}},
    //             {3, {tensor3, "matmul0_output"}},
    //             {4, {tensor4, "matmul1_output"}},
    //             {5, {tensor5, "output"}},
    //         },
    //     };
    // }

    // TEST(Graph, MutantGenerator) {
    //     auto graphTopo = TestInGraphBuild().build();
    //     fmt::println("{}", graphTopo.topology.toString());
    //     GraphMutant g(std::move(graphTopo));
    //     // create mutant generator
    //     MutantGenerator mutant;
    //     OpVec oplist = {std::make_shared<MatMulBox>(), std::make_shared<ConcatBox>()};
    //     mutant.init(1.0, 3, oplist);
    //     std::vector<GraphMutant> outGraph = {};
    //     mutant.run(std::move(g), outGraph);
    //     // for (size_t i = 0; i < outGraph.size(); ++i) {
    //     //     fmt::println("{}", outGraph[i].internal().toString());
    //     // }
    // }

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestInGraphBuild1() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_shared<ConvBox>(), "conv"};

        auto tensor0 = Tensor::share(DataType::F32, {1, 3, 3, 3}, LayoutType::NCHW);
        auto tensor1 = Tensor::share(DataType::F32, {2, 3, 1, 1}, LayoutType::NCHW);
        auto tensor2 = Tensor::share(DataType::F32, {1, 2, 3, 3}, LayoutType::NCHW);
        auto weight = reinterpret_cast<float *>(tensor1->malloc());
        auto input = reinterpret_cast<float *>(tensor0->malloc());
        std::iota(weight, weight + tensor1->elementsSize(), 1.0);
        std::iota(input, input + tensor0->elementsSize(), 1.0);
        float outputData[]{78.0, 84.0, 90.0, 96.0, 102.0, 108.0, 114.0, 120.0, 126.0, 168.0, 183.0, 198.0,
                           213.0, 228.0, 243.0, 258.0, 273.0, 288.0};
        std::memcpy(tensor2->malloc(), outputData, tensor2->bytesSize());

        return {
            {
                {0, {{0, 1}, {2}}},
            },
            {0, 1},
            {2},
            std::move(nodes),
            {
                {0, {tensor0, "input"}},
                {1, {tensor1, "weight"}},
                {2, {tensor2, "output"}},
            },
        };
    }

    TEST(Graph, MutantGeneratorConv1x1) {
        auto graphTopo1 = TestInGraphBuild1().build();
        fmt::println("{}", graphTopo1.topology.toString());
        GraphMutant g(std::move(graphTopo1));
        // create mutant generator
        MutantGenerator mutant;
        Shape perm1 = {0, 2, 3, 1};
        Shape perm2 = {1, 0, 2, 3};
        Shape perm3 = {0, 3, 1, 2};
        Shape shape1 = {-1, 0};
        Shape shape2 = {0, -1};
        Shape shape3 = {1, 3, 3, 2};
        OpVec oplist = {
            std::make_shared<MatMulBox>(),
            std::make_shared<TransposeBox>(perm1),
            std::make_shared<TransposeBox>(perm2),
            std::make_shared<TransposeBox>(perm3),
            std::make_shared<ReshapeBox>(shape1),
            std::make_shared<ReshapeBox>(shape2),
            std::make_shared<ReshapeBox>(shape3),
        };
        mutant.init(1.0, 8, oplist);
        std::vector<GraphMutant> outGraph = {};
        mutant.run(std::move(g), outGraph);
        fmt::println("{}", outGraph.size());
    }
}// namespace refactor::computation