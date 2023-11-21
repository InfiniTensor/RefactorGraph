#ifdef USE_CUDA

#include "../src/kernels/simple_binary/basic_cpu.hh"
#include "../src/kernels/simple_binary/binary_cudnn.hh"
#include "../src/kernels/simple_binary/no_broadcast_cpu.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCudnn(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    auto aTensor = Tensor::share(DataType::F32, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::F32, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::F32, dimC, LayoutType::NCHW);
    auto kernel = BinaryCudnn::build(binaryOPT, *aTensor, *bTensor, *cTensor);
    auto kCpu = dimA == dimB
                    ? Binary11Cpu::build(binaryOPT, *aTensor, *bTensor)
                    : BinaryBasicCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res),
         rCpu = kCpu->lower(res);
    // Init inputs and outputs
    std::vector<float>
        a(aTensor->elementsSize(), 3.0f),
        b(bTensor->elementsSize(), 2.0f),
        c(cTensor->elementsSize());
    auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), aTensor->bytesSize()),
         bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), bTensor->bytesSize()),
         cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), cTensor->bytesSize());
    aGPU->copyIn(a.data(), aTensor->bytesSize());
    bGPU->copyIn(b.data(), bTensor->bytesSize());
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
    cGPU->copyOut(result.data(), cTensor->bytesSize());
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
