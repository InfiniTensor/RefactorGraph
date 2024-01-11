#ifdef USE_CUDA

#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include "../src/kernels/simple_binary/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

template<decltype(DataType::internal) T>
void testBinaryCuda(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    using T_ = primitive<T>::type;
    auto aTensor = Tensor::share(T, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(T, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(T, dimC, LayoutType::NCHW);

    auto cpuKernel = BinaryCpu::build(binaryOPT, *aTensor, *bTensor),
         cudaKernel = BinaryCuda::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel && cudaKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    auto cudaRoutine = cudaKernel->lower(res).routine;

    // Init inputs and outputs
    std::vector<T_> a(aTensor->elementsSize(), 3);
    std::vector<T_> b(bTensor->elementsSize(), 2);
    std::vector<T_> c(cTensor->elementsSize());
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
        cudaRoutine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{a.data(), b.data()};
        void *outputs[]{c.data()};
        cpuRoutine(res, nullptr, inputs, outputs);
    }
    // Compare
    std::vector<T_> result(cTensor->elementsSize());
    cGPU->copyToHost(result.data(), cTensor->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_EQ(c[i], result[i]);
    }
}

TEST(kernel, BinaryCudaAdd) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Add,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaMul) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Mul,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaSub) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Sub,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaDiv) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Div,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaMod) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Mod,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaFmodI8) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Fmod,
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4},
                                 Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaFmodF32) {
    testBinaryCuda<DataType::F32>(SimpleBinaryType::Fmod,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaBroadcast) {
    testBinaryCuda<DataType::I8>(SimpleBinaryType::Add, Shape{1, 2, 3, 4, 5, 6}, Shape{}, Shape{1, 2, 3, 4, 5, 6});
}

#endif
