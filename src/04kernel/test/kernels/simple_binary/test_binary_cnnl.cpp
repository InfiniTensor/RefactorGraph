#ifdef USE_BANG

#include "../src/kernels/simple_binary/binary_cnnl.hh"
#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testBinaryCnnl(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, dimC, LayoutType::NCHW);
    auto kernel = BinaryCnnl::build(binaryOPT, *aTensor, *bTensor, *cTensor);
    auto kCpu = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    auto rCpu = kCpu->lower(res).routine;
    // Init inputs and outputs
    std::vector<float>
        a(aTensor->elementsSize(), 3.0f),
        b(bTensor->elementsSize(), 2.0f),
        c(cTensor->elementsSize());
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         aMLU = dev.malloc(aTensor->bytesSize()),
         bMLU = dev.malloc(bTensor->bytesSize()),
         cMLU = dev.malloc(cTensor->bytesSize());
    aMLU->copyFromHost(a.data(), aTensor->bytesSize());
    bMLU->copyFromHost(b.data(), bTensor->bytesSize());
    // Compute
    {
        void const *inputs[]{*aMLU, *bMLU};
        void *outputs[]{*cMLU};
        routine(res, *workspace, inputs, outputs);
    }
    {
        void const *inputs[]{a.data(), b.data()};
        void *outputs[]{c.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // Compare
    std::vector<float> result(cTensor->elementsSize());
    cMLU->copyToHost(result.data(), cTensor->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(c[i], result[i]);
    }
}

TEST(kernel, BinaryCnnlAdd) {
    testBinaryCnnl(SimpleBinaryType::Add, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCnnlMul) {
    testBinaryCnnl(SimpleBinaryType::Mul, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCnnlSub) {
    testBinaryCnnl(SimpleBinaryType::Sub, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCnnlDiv) {
    testBinaryCnnl(SimpleBinaryType::Div, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

// TEST(kernel, BinaryCnnlAnd) {
//     testBinaryCnnl(SimpleBinaryType::And, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
// }

// TEST(kernel, BinaryCnnlOr) {
//     testBinaryCnnl(SimpleBinaryType::Or, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
// }

// TEST(kernel, BinaryCnnlXor) {
//     testBinaryCnnl(SimpleBinaryType::Xor, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
// }

TEST(kernel, BinaryCnnlPow) {
    testBinaryCnnl(SimpleBinaryType::Pow, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40}, Shape{10, 20, 30, 40});
}

TEST(kernel, BinaryCnnlBroadcast) {
    testBinaryCnnl(SimpleBinaryType::Add, Shape{3, 4, 5, 6}, Shape{}, Shape{3, 4, 5, 6});
}

#endif
