#include "../../../src/kernels/select/cpu_kernel.hh"
#include <functional>
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

static void testSelect(const SelectType selectType, const std::vector<Shape> &shapes, const std::vector<std::vector<float>> &data,
                       const std::vector<float> expectData) {
    // build routine
    TensorRefs dataTensors;
    std::vector<Tensor> tensorsVec;
    for (size_t i = 0; i < shapes.size(); ++i) {
        tensorsVec.push_back(Tensor(DataType::F32, shapes[i], LayoutType::Others, nullptr));
    }
    for (size_t i = 0; i < shapes.size(); ++i) {
        dataTensors.push_back(std::cref(tensorsVec[i]));
    }
    auto kernel = SelectCpu::build(selectType, dataTensors);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input output data
    void const *inputs[data.size()];
    for (size_t i = 0; i < data.size(); ++i) {
        inputs[i] = data[i].data();
    }
    std::vector<float>
        out(expectData.size());
    void *outputs[]{out.data()};
    // inference
    routine(res, nullptr, inputs, outputs);
    // check
    for (auto i : range0_(expectData.size())) {
        EXPECT_FLOAT_EQ(expectData[i], out[i]);
    }
}

TEST(kernel, SelectCpu) {
    // no need broadcast
    testSelect(SelectType::Max,
               {{1, 3}, {1, 3}, {1, 3}},
               {{3, 2, 1}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{1, 3}, {1, 3}, {1, 3}},
               {{3, 2, 1}, {1, 4, 4}, {2, 5, 3}},
               {1, 2, 1});

    // need broadcast
    testSelect(SelectType::Max,
               {{3}, {1, 3}, {1, 3}},
               {{3, 3, 3}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{3}, {1, 3}, {1, 3}},
               {{3, 3, 3}, {1, 4, 4}, {2, 5, 3}},
               {1, 3, 3});

    testSelect(SelectType::Max,
               {{1}, {1, 3}, {1, 3}},
               {{3}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{1}, {1, 3}, {1, 3}},
               {{3}, {1, 4, 4}, {2, 5, 3}},
               {1, 3, 3});
}
