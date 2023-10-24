#include "../src/kernels/simple_binary/arthimetic11.hh"
#include "kernel/tensor.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCPU(SimpleBinaryType binaryOPT, std::function<float(float, float)> operation) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = Arthimetic11::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel);
    auto cpuRoutine = cpuKernel->lower();

    // Init inputs and outputs
    std::vector<float> a(aTensor->elementsSize(), 3.0f);
    std::vector<float> b(bTensor->elementsSize(), 2.0f);
    std::vector<float> c(cTensor->elementsSize());

    // Compute
    auto res = runtime::Resources();
    void const *inputsCPU[]{a.data(), b.data()};
    void *outputsCPU[]{c.data()};
    cpuRoutine(res, inputsCPU, outputsCPU);

    // Compare
    for (auto i : range0_(c.size())) {
        EXPECT_FLOAT_EQ(c[i], operation(a[i], b[i]));
    }
}

TEST(kernel, BinaryCpu) {
    testBinaryCPU(SimpleBinaryType::Add, [](float a, float b) { return a + b; });
    testBinaryCPU(SimpleBinaryType::Sub, [](float a, float b) { return a - b; });
    testBinaryCPU(SimpleBinaryType::Mul, [](float a, float b) { return a * b; });
    testBinaryCPU(SimpleBinaryType::Div, [](float a, float b) { return a / b; });
}
