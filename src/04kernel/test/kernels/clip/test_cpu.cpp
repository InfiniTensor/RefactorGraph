#include "../../../src/kernels/clip/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, ClipCpu) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{2, 3, 4, 5});
    auto kernel = ClipCpu::build(*data, true);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> value(data->elementsSize());
    float min = 30, max = 80;
    std::iota(value.begin(), value.end(), 0);
    // inference
    {
        void const *inputs[]{value.data(), &min, &max};
        void *outputs[]{value.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(static_cast<size_t>(min))) {
        EXPECT_EQ(value[i], min);
    }
    for (auto i : range(static_cast<size_t>(min), static_cast<size_t>(max))) {
        EXPECT_EQ(value[i], i);
    }
    for (auto i : range(static_cast<size_t>(max), data->elementsSize())) {
        EXPECT_EQ(value[i], max);
    }
}
