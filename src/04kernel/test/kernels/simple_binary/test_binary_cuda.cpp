#ifdef USE_CUDA

#include "../src/kernels/simple_binary/basic_cpu.hh"
#include "../src/kernels/simple_binary/basic_cuda.hh"
#include "../src/kernels/simple_binary/no_broadcast_cpu.hh"
#include "../src/kernels/simple_binary/no_broadcast_cuda.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

void testBinaryCuda(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    using T_ = primitive<DataType::I8>::type;
    auto aTensor = Tensor::share(DataType::I8, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(DataType::I8, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(DataType::I8, dimC, LayoutType::NCHW);

    auto cpuKernel = dimA == dimB
                         ? Binary11Cpu::build(binaryOPT, *aTensor, *bTensor)
                         : BinaryBasicCpu::build(binaryOPT, *aTensor, *bTensor);
    auto cudaKernel = dimA == dimB
                          ? Binary11Cuda::build(binaryOPT, *aTensor, *bTensor)
                          : BinaryBasicCuda::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(cpuKernel && cudaKernel);
    auto res = runtime::Resources();
    auto cpuRoutine = cpuKernel->lower(res).routine;
    auto cudaRoutine = cudaKernel->lower(res).routine;

    // Init inputs and outputs
    std::vector<T_> a(aTensor->elementsSize(), 3.0f);
    std::vector<T_> b(bTensor->elementsSize(), 2.0f);
    std::vector<T_> c(cTensor->elementsSize());
    auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), aTensor->bytesSize());
    auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), bTensor->bytesSize());
    auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), cTensor->bytesSize());
    aGPU->copyIn(a.data(), aTensor->bytesSize());
    bGPU->copyIn(b.data(), bTensor->bytesSize());
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
    cGPU->copyOut(result.data(), cTensor->bytesSize());
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
