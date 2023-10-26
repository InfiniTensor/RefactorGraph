#ifdef USE_CUDA
#include "../src/kernels/gather/cpu_kernel.hh"
#include "../src/kernels/gather/cuda_kernel.hh"
#include <gtest/gtest.h>
using namespace refactor;
using namespace kernel;

TEST(kernel, GatherCuda) {
    // Case axis = 0, indexType= int64
    {
        // Create Tensor and build kernels
        auto aTensor = Tensor::share(DataType::F32, Shape{3, 2}, LayoutType::NCHW);
        auto bTensor = Tensor::share(DataType::I64, Shape{2, 2}, LayoutType::NCHW);
        auto cTensor = Tensor::share(DataType::F32, Shape{2, 2, 2}, LayoutType::NCHW);
        uint32_t axis = 0;
        auto cpuKernel = GatherCpu::build(GatherInfo(axis, *aTensor, *bTensor));
        ASSERT_TRUE(cpuKernel);
        auto cpuRoutine = cpuKernel->lower();
        auto cudaKernel = GatherCuda::build(*aTensor, *bTensor, *cTensor, axis);
        ASSERT_TRUE(cudaKernel);
        auto cudaRoutine = cudaKernel->lower();

        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 2.3, 3.4, 4.5, 5.7};
        std::vector<int64_t> b{0, 1, 1, 2};
        std::vector<float> c(cTensor->elementsSize());
        auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), aTensor->bytesSize());
        auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), bTensor->bytesSize());
        auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), cTensor->bytesSize());
        aGPU->copyIn(a.data(), aTensor->bytesSize());
        bGPU->copyIn(b.data(), bTensor->bytesSize());

        // Compute
        auto res = runtime::Resources();
        void const *inputsCPU[]{a.data(), b.data()};
        void *outputsCPU[]{c.data()};
        cpuRoutine(res, inputsCPU, outputsCPU);
        void const *inputsGPU[]{*aGPU, *bGPU};
        void *outputsGPU[]{*cGPU};
        cudaRoutine(res, inputsGPU, outputsGPU);

        // Compare
        std::vector<float> result(cTensor->elementsSize());
        cGPU->copyOut(result.data(), cTensor->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }

    // Case axis = 1, indexType= int32
    {
        // Create Tensor and build kernels
        auto aTensor = Tensor::share(DataType::F32, Shape{3, 3}, LayoutType::NCHW);
        auto bTensor = Tensor::share(DataType::I32, Shape{1, 2}, LayoutType::NCHW);
        auto cTensor = Tensor::share(DataType::F32, Shape{3, 1, 2}, LayoutType::NCHW);
        uint32_t axis = 1;
        auto cpuKernel = GatherCpu::build(GatherInfo(axis, *aTensor, *bTensor));
        ASSERT_TRUE(cpuKernel);
        auto cpuRoutine = cpuKernel->lower();
        auto cudaKernel = GatherCuda::build(*aTensor, *bTensor, *cTensor, axis);
        ASSERT_TRUE(cudaKernel);
        auto cudaRoutine = cudaKernel->lower();

        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9};
        std::vector<int> b{0, 2};
        std::vector<float> c(cTensor->elementsSize());
        auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), aTensor->bytesSize());
        auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), bTensor->bytesSize());
        auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), cTensor->bytesSize());
        aGPU->copyIn(a.data(), aTensor->bytesSize());
        bGPU->copyIn(b.data(), bTensor->bytesSize());

        // Compute
        auto res = runtime::Resources();
        void const *inputsCPU[]{a.data(), b.data()};
        void *outputsCPU[]{c.data()};
        cpuRoutine(res, inputsCPU, outputsCPU);
        void const *inputsGPU[]{*aGPU, *bGPU};
        void *outputsGPU[]{*cGPU};
        cudaRoutine(res, inputsGPU, outputsGPU);

        // Compare
        std::vector<float> result(cTensor->elementsSize());
        cGPU->copyOut(result.data(), cTensor->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }
}
#endif
