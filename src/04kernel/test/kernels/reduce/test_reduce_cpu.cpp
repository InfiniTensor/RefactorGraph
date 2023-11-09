#include "../../../src/kernels/reduce/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

static void testReducemean(const Shape &shape, const std::vector<float> &data,
                           Axes axes, const std::vector<float> ExpectData) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, shape);
    auto kernel = ReduceCpu::build(axes, ReduceType::Mean, {*dataTensor});
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // put input output data
    void const *inputs[]{data.data()};
    std::vector<float>
        out(data.size());
    void *outputs[]{out.data()};
    // inference
    routine(res, inputs, outputs);
    // check
    for (auto i : range0_(ExpectData.size())) {
        EXPECT_FLOAT_EQ(ExpectData[i], out[i]);
    }
}

TEST(kernel, ReduceMeanCpu) {
    testReducemean({2, 3, 2, 2},
                   {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23},
                   {1, 2},
                   {5, 6, 17, 18});
    testReducemean({2, 3, 2, 2, 1},
                   {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23},
                   {1, 2},
                   {5, 6, 17, 18});
}
