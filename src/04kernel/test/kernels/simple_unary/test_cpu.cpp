#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

using VecFloat = std::vector<float>;

static void testOp(SimpleUnaryType opType, float check(float)) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    VecFloat data(dataTensor->elementsSize());
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

static void testOpWithData(SimpleUnaryType opType, const VecFloat &data) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3});
    auto kernel = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    VecFloat inputdata(dataTensor->elementsSize());
    for (auto i : range0_(inputdata.size())) { inputdata[i] = i; }
    auto result = inputdata;
    // inference
    {
        void const *inputs[]{result.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(inputdata.size())) {
        EXPECT_NEAR(data[i], result[i], 1e-5);
    }
}

TEST(kernel, SimpleUnaryCpu) {
    testOp(SimpleUnaryType::Abs, std::abs);
    testOp(SimpleUnaryType::Sqrt, std::sqrt);
    testOp(SimpleUnaryType::Tanh, std::tanh);
    testOp(SimpleUnaryType::Erf, std::erf);
    testOpWithData(SimpleUnaryType::HardSwish,
                   VecFloat{0.000000, 0.666667, 1.666667, 3.000000, 4.000000, 5.000000});
    testOpWithData(SimpleUnaryType::Exp, VecFloat{1.000000, 2.718282, 7.389056, 20.085537, 54.598148, 148.413162});
}
