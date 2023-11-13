#include "computation/graph_mutant.h"
#include "computation/mutant_generator.h"
#include "computation/operators/concat.h"
#include "computation/operators/mat_mul.h"
#include <gtest/gtest.h>
#include <numeric>

namespace refactor::computation {

    refactor::graph_topo::Builder<size_t, Node, size_t, Edge> TestInGraphBuild() {
        auto nodes = std::unordered_map<size_t, Node>{};
        nodes[0] = Node{std::make_shared<MatMulBox>(), "matmul_1"};
        nodes[1] = Node{std::make_shared<MatMulBox>(), "matmul_2"};
        nodes[2] = Node{std::make_shared<ConcatBox>(), "concat"};

        auto tensor0 = Tensor::share(DataType::F32, {5, 6}, LayoutType::Others);
        auto tensor1 = Tensor::share(DataType::F32, {4, 5}, LayoutType::Others);
        auto tensor2 = Tensor::share(DataType::F32, {5, 7}, LayoutType::Others);
        auto tensor3 = Tensor::share(DataType::F32, {4, 6}, LayoutType::Others);
        auto tensor4 = Tensor::share(DataType::F32, {4, 7}, LayoutType::Others);
        auto tensor5 = Tensor::share(DataType::F32, {4, 13}, LayoutType::Others);
        // initialize inputs data
        auto data0 = reinterpret_cast<float *>(tensor0->malloc());
        auto data1 = reinterpret_cast<float *>(tensor1->malloc());
        auto data2 = reinterpret_cast<float *>(tensor2->malloc());
        std::iota(data0, data0 + tensor0->elementsSize(), 1.0);
        std::iota(data1, data1 + tensor1->elementsSize(), 1.0);
        std::iota(data2, data2 + tensor2->elementsSize(), 1.0);
        // initialize outputs data
        float outputData[]{255.0, 270.0, 285.0, 300.0, 315.0, 330.0, 295.0, 310.0, 325.0, 340.0, 355.0, 370.0, 385.0, 580.0, 620.0, 660.0, 700.0,
                           740.0, 780.0, 670.0, 710.0, 750.0, 790.0, 830.0, 870.0, 910.0, 905.0, 970.0, 1035.0, 1100.0, 1165.0, 1230.0, 1045.0, 1110.0, 1175.0, 1240.0,
                           1305.0, 1370.0, 1435.0, 1230.0, 1320.0, 1410.0, 1500.0, 1590.0, 1680.0, 1420.0, 1510.0, 1600.0, 1690.0, 1780.0, 1870.0, 1960.};
        std::memcpy(tensor5->malloc(), outputData, tensor5->bytesSize());

        return {
            {{0, {{1, 0}, {3}}},
             {1, {{1, 2}, {4}}},
             {2, {{3, 4}, {5}}}},
            {0, 1, 2},
            {5},
            std::move(nodes),
            {
                {0, {tensor0, "input_tensor_0"}},
                {1, {tensor1, "input_tensor_1"}},
                {2, {tensor2, "input_tensor_2"}},
                {3, {tensor3, "matmul0_output"}},
                {4, {tensor4, "matmul1_output"}},
                {5, {tensor5, "output"}},
            },
        };
    }

    TEST(Graph, MutantGenerator) {
        auto graphTopo = TestInGraphBuild().build();
        fmt::println("{}", graphTopo.topology.toString());
        GraphMutant g(std::move(graphTopo));
        // create mutant generator
        MutantGenerator mutant;
        OpVec oplist = {std::make_shared<MatMulBox>(), std::make_shared<ConcatBox>()};
        mutant.init(1.0, 3, oplist);
        std::vector<GraphMutant> outGraph = {};
        mutant.run(std::move(g), outGraph);
        // for (size_t i = 0; i < outGraph.size(); ++i) {
        //     fmt::println("{}", outGraph[i].internal().toString());
        // }
    }
}// namespace refactor::computation