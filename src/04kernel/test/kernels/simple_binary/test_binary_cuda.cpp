#ifdef USE_CUDA

#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include "../src/kernels/simple_binary/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testBinaryCuda(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    using T_ = primitive<DataType::I8>::type;
    auto aTensor = Tensor::share(DataType::I8, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::I8, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::I8, dimC, LayoutType::NCHW);

    auto cpuKernel = BinaryCpu::build(binaryOPT, *aTensor, *bTensor),
         cudaKernel = BinaryCuda::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel && cudaKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    auto cudaRoutine = cudaKernel->lower(res).routine;

    // Init inputs and outputs
    std::vector<T_> a(aTensor->elementsSize(), 3.0f);
    std::vector<T_> b(bTensor->elementsSize(), 2.0f);
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
    testBinaryCuda(SimpleBinaryType::Add,
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaMul) {
    testBinaryCuda(SimpleBinaryType::Mul,
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaSub) {
    testBinaryCuda(SimpleBinaryType::Sub,
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaDiv) {
    testBinaryCuda(SimpleBinaryType::Div,
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4},
                   Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCudaBroadcast) {
    testBinaryCuda(SimpleBinaryType::Add, Shape{1, 2, 3, 4, 5, 6}, Shape{}, Shape{1, 2, 3, 4, 5, 6});
}

#endif
