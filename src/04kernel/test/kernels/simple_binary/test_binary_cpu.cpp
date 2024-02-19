#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include <cmath>
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCPU(SimpleBinaryType binaryOPT, std::function<float(float, float)> operation) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    // Init inputs and outputs
    std::vector<float> a(aTensor->elementsSize(), 3.0f);
    std::vector<float> b(bTensor->elementsSize(), 2.0f);
    std::vector<float> c(cTensor->elementsSize());
    // Compute
    void const *inputs[]{a.data(), b.data()};
    void *outputs[]{c.data()};
    cpuRoutine(res, nullptr, inputs, outputs);
    // Compare
    for (auto i : range0_(c.size())) {
        EXPECT_FLOAT_EQ(c[i], operation(a[i], b[i]));
    }
}

void testModCPU(SimpleBinaryType binaryOPT, std::function<int(int, int)> operation) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    // Init inputs and outputs
    std::vector<int> a(aTensor->elementsSize(), -3);
    std::vector<int> b(bTensor->elementsSize(), 2);
    std::vector<int> c(cTensor->elementsSize());
    // Compute
    void const *inputs[]{a.data(), b.data()};
    void *outputs[]{c.data()};
    cpuRoutine(res, nullptr, inputs, outputs);
    // Compare
    for (auto i : range0_(c.size())) {
        EXPECT_FLOAT_EQ(c[i], operation(a[i], b[i]));
    }
}

void testFmodWithI32CPU(SimpleBinaryType binaryOPT, std::function<int(int, int)> operation) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::I32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    // Init inputs and outputs
    std::vector<int> a(aTensor->elementsSize(), -3);
    std::vector<int> b(bTensor->elementsSize(), 2);
    std::vector<int> c(cTensor->elementsSize());
    // Compute
    void const *inputs[]{a.data(), b.data()};
    void *outputs[]{c.data()};
    cpuRoutine(res, nullptr, inputs, outputs);
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
    testModCPU(SimpleBinaryType::Mod, [](int a, int b) { return a % b; });
    testFmodWithI32CPU(SimpleBinaryType::Fmod, [](int a, int b) { return static_cast<int32_t>(std::fmod(a, b)); });
    testBinaryCPU(SimpleBinaryType::Fmod, [](float a, float b) { return std::fmod(a, b); });
}

TEST(kernel, BinaryCpuBroadcast) {
    // build routine
    auto a = Tensor::share(DataType::F32, Shape{20, 30, 1});
    auto b = Tensor::share(DataType::F32, Shape{30, 50});
    auto c = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = BinaryCpu::build(SimpleBinaryType::Add, *a, *b);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        dataA(a->elementsSize()),
        dataB(b->elementsSize()),
        dataC(c->elementsSize());
    for (auto i : range0_(dataA.size())) { dataA[i] = 11; }
    for (auto i : range0_(dataB.size())) { dataB[i] = 7; }
    // inference
    {
        void const *inputs[]{dataA.data(), dataB.data()};
        void *outputs[]{dataC.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto x : dataC) {
        EXPECT_FLOAT_EQ(18, x);
    }
}
