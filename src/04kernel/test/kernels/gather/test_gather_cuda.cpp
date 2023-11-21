#ifdef USE_CUDA

#include "../src/kernels/gather/cpu_kernel.hh"
#include "../src/kernels/gather/cuda_kernel.hh"
#include "kernel/target.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, GatherCuda) {
    // Case axis = 0, indexType= int64
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 2}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I64, Shape{2, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{2, 2, 2}, LayoutType::NCHW);
        GatherInfo info(0, *data, *indices);
        auto cudaKernel = GatherCuda::build(info);
        auto cpuKernel = GatherCpu::build(info);
        ASSERT_TRUE(cudaKernel && cpuKernel);
        auto res = runtime::Resources();
        auto cudaRoutine = cudaKernel->lower(res).routine;
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 2.3, 3.4, 4.5, 5.7};
        std::vector<int64_t> b{0, 1, 1, 2};
        std::vector<float> c(output->elementsSize());
        auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), data->bytesSize());
        auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), indices->bytesSize());
        auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), output->bytesSize());
        aGPU->copyIn(a.data(), data->bytesSize());
        bGPU->copyIn(b.data(), indices->bytesSize());
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
        std::vector<float> result(output->elementsSize());
        cGPU->copyOut(result.data(), output->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }

    // Case axis = 1, indexType= int32
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 3}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I32, Shape{1, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{3, 1, 2}, LayoutType::NCHW);
        GatherInfo info(1, *data, *indices);
        auto cudaKernel = GatherCuda::build(info);
        auto cpuKernel = GatherCpu::build(info);
        ASSERT_TRUE(cudaKernel && cpuKernel);
        auto res = runtime::Resources();
        auto cudaRoutine = cudaKernel->lower(res).routine;
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9};
        std::vector<int> b{0, 2};
        std::vector<float> c(output->elementsSize());
        auto aGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), data->bytesSize());
        auto bGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), indices->bytesSize());
        auto cGPU = mem_manager::ForeignBlob::share(Target(Target::NvidiaGpu).memManager(), output->bytesSize());
        aGPU->copyIn(a.data(), data->bytesSize());
        bGPU->copyIn(b.data(), indices->bytesSize());
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
        std::vector<float> result(output->elementsSize());
        cGPU->copyOut(result.data(), output->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }
}

#endif
