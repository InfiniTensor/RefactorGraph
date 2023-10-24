#ifdef USE_CUDA
#include "../src/kernels/simple_binary/arthimetic11.hh"
#include "../src/kernels/simple_binary/binary_cudnn.hh"
#include "kernel/tensor.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCudnn(SimpleBinaryType binaryOPT) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, Shape{10, 20, 30, 40}, LayoutType::NCHW);
    auto cpuKernel = Arthimetic11::build(binaryOPT, *aTensor, *bTensor);
    auto cudnnKernel = BinaryCudnn::build(binaryOPT, *aTensor, *bTensor, *cTensor);
    ASSERT_TRUE(cpuKernel);
    ASSERT_TRUE(cudnnKernel);
    auto cpuRoutine = cpuKernel->lower();
    auto cudnnRoutine = cudnnKernel->lower();

    // Init inputs and outputs
    std::vector<float> a(aTensor->elementsSize(), 3.0f);
    std::vector<float> b(bTensor->elementsSize(), 2.0f);
    std::vector<float> c(cTensor->elementsSize());
    auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memFunc(), aTensor->bytesSize());
    auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memFunc(), bTensor->bytesSize());
    auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memFunc(), cTensor->bytesSize());
    aGPU->copyIn(a.data(), aTensor->bytesSize());
    bGPU->copyIn(b.data(), bTensor->bytesSize());

    // Compute
    auto res = runtime::Resources();
    void const *inputsGPU[]{*aGPU, *bGPU};
    void *outputsGPU[]{*cGPU};
    cudnnRoutine(res, inputsGPU, outputsGPU);
    void const *inputsCPU[]{a.data(), b.data()};
    void *outputsCPU[]{c.data()};
    cpuRoutine(res, inputsCPU, outputsCPU);

    // Compare
    std::vector<float> result(cTensor->elementsSize());
    cGPU->copyOut(result.data(), cTensor->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(c[i], result[i]);
    }
}

TEST(kernel, BinaryCudnnAdd) {
    testBinaryCudnn(SimpleBinaryType::Add);
}

TEST(kernel, BinaryCudnnMul) {
    testBinaryCudnn(SimpleBinaryType::Mul);
}

TEST(kernel, BinaryCudnnSub) {
    testBinaryCudnn(SimpleBinaryType::Sub);
}

#endif
