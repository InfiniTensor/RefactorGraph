﻿#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

static void testOp(SimpleUnaryType opType, float check(float)) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
    auto result = data;
    // inference
    {
        void const *inputs[]{result.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(check(data[i]), result[i]);
    }
}

TEST(kernel, SimpleUnaryCpu) {
    testOp(SimpleUnaryType::Abs, std::abs);
    testOp(SimpleUnaryType::Sqrt, std::sqrt);
    testOp(SimpleUnaryType::Tanh, std::tanh);
}
