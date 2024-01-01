#include "../../../src/kernels/dynamic_quantize_linear/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, DynamicQuantizeLinearCpu) {
    // build routine
    auto kernel = DynamicQuantizeLinearCpu::build(6);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> x{0, 2, -3, -2.5, 1.34, 0.5};
    std::vector<uint8_t> y(x.size());
    float scale;
    uint8_t zeroPoint;
    // inference
    {
        void const *inputs[]{x.data()};
        void *outputs[]{y.data(), &scale, &zeroPoint};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    EXPECT_FLOAT_EQ(scale, (2 + 3) / 255.f);
    EXPECT_EQ(zeroPoint, 153);
    for (auto i : range0_(y.size())) {
        EXPECT_EQ(y[i], static_cast<uint8_t>(std::round(x[i] / scale) + zeroPoint));
    }
}
