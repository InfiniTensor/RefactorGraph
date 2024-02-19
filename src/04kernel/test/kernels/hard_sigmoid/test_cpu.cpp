#include "../../../src/kernels/hard_sigmoid/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, HardSigmoidCpu) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 5});
    float alpha = 0.2f, beta = 0.5f;
    auto kernel = HardSigmoidCpu::build(alpha, beta, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> result(dataTensor->elementsSize());
    for (auto i : range0_(result.size())) { result[i] = i; }
    // inference
    {
        void const *inputs[]{result.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    std::vector<float> output = {0.5, 0.7, 0.9, 1., 1., 1., 1., 1., 1.,
                                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                 1., 1., 1., 1., 1., 1.};
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(output[i], result[i]);
    }
}
