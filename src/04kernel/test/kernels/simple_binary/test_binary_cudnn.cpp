#ifdef USE_CUDA

#include "../src/kernels/simple_binary/binary_cudnn.hh"
#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testBinaryCudnn(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, dimC, LayoutType::NCHW);
    auto kernel = BinaryCudnn::build(binaryOPT, *aTensor, *bTensor, *cTensor);
    auto kCpu = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // Init inputs and outputs
    std::vector<float>
        a(aTensor->elementsSize(), 3.0f),
        b(bTensor->elementsSize(), 2.0f),
        c(cTensor->elementsSize());
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto aGPU = dev.malloc(aTensor->bytesSize()),
         bGPU = dev.malloc(bTensor->bytesSize()),
         cGPU = dev.malloc(cTensor->bytesSize());
    aGPU->copyFromHost(a.data(), aTensor->bytesSize());
    bGPU->copyFromHost(b.data(), bTensor->bytesSize());
    // Compute
    {
        void const *inputs[]{*aGPU, *bGPU};
        void *outputs[]{*cGPU};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{a.data(), b.data()};
        void *outputs[]{c.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // Compare
    std::vector<float> result(cTensor->elementsSize());
    cGPU->copyToHost(result.data(), cTensor->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(c[i], result[i]);
    }
}

TEST(kernel, BinaryCudnnAdd) {
    testBinaryCudnn(SimpleBinaryType::Add, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCudnnMul) {
    testBinaryCudnn(SimpleBinaryType::Mul, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCudnnSub) {
    testBinaryCudnn(SimpleBinaryType::Sub, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCudnnBroadcast) {
    testBinaryCudnn(SimpleBinaryType::Add, Shape{3, 4, 5, 6}, Shape{}, Shape{3, 4, 5, 6});
}

#endif
