#include "../src/kernels/simple_binary/basic_cpu.hh"
#include "../src/kernels/simple_binary/no_broadcast_cpu.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCPU(SimpleBinaryType binaryOPT, std::function<float(float, float)> operation) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = Binary11Cpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res);
    // Init inputs and outputs
    std::vector<float> a(aTensor->elementsSize(), 3.0f);
    std::vector<float> b(bTensor->elementsSize(), 2.0f);
    std::vector<float> c(cTensor->elementsSize());
    // Compute
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

TEST(kernel, BinaryCpuBroadcast) {
    // build routine
    auto a = Tensor::share(DataType::F32, Shape{20, 30, 1});
    auto b = Tensor::share(DataType::F32, Shape{30, 50});
    auto c = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = BinaryBasicCpu::build(SimpleBinaryType::Add, *a, *b);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto ma = mem_manager::ForeignBlob::share(mfn, a->bytesSize());
    auto mb = mem_manager::ForeignBlob::share(mfn, b->bytesSize());
    auto mc = mem_manager::ForeignBlob::share(mfn, c->bytesSize());
    // put input data
    std::vector<float> data(a->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 11; }
    ma->copyIn(data.data(), a->bytesSize());
    data.resize(b->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 7; }
    mb->copyIn(data.data(), b->bytesSize());
    // inference
    void const *inputs[]{*ma, *mb};
    void *outputs[]{*mc};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(c->elementsSize());
    mc->copyOut(result.data(), c->bytesSize());
    // check
    for (auto x : result) {
        EXPECT_FLOAT_EQ(18, x);
    }
}
