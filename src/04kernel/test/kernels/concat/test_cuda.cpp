﻿#ifdef USE_CUDA

#include "../../../src/kernels/concat/cpu_kernel.hh"
#include "../../../src/kernels/concat/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ConcatCuda) {
    // build routine
    std::vector<Arc<Tensor>> inputTensors{
        Tensor::share(DataType::F32, Shape{2, 3, 1, 1, 7, 7}),// 勿
        Tensor::share(DataType::F32, Shape{2, 3, 1, 9, 7, 7}),// 忘
        Tensor::share(DataType::F32, Shape{2, 3, 1, 3, 7, 7}),// 国
        Tensor::share(DataType::F32, Shape{2, 3, 1, 7, 7, 7}),// 耻
    };
    auto result = Tensor::share(DataType::F32, Shape{2, 3, 1, 20, 7, 7});
    TensorRefs inputs_;
    inputs_.reserve(inputTensors.size());
    std::transform(inputTensors.begin(), inputTensors.end(),
                   std::back_inserter(inputs_),
                   [](auto const &it) { return std::cref(*it); });
    SplitInfo info(3, inputs_);
    auto kCpu = ConcatCpu::build(info);
    auto kernel = ConcatCuda::build(info);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto routine = kernel->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    Arc<Device::Blob>
        gpuIns[]{
            dev.malloc(inputTensors[0]->bytesSize()),
            dev.malloc(inputTensors[1]->bytesSize()),
            dev.malloc(inputTensors[2]->bytesSize()),
            dev.malloc(inputTensors[3]->bytesSize()),
        },
        gpuOut = dev.malloc(result->bytesSize());
    // put input data
    std::vector<float>
        cpuIns[]{
            std::vector<float>(inputTensors[0]->elementsSize()),
            std::vector<float>(inputTensors[1]->elementsSize()),
            std::vector<float>(inputTensors[2]->elementsSize()),
            std::vector<float>(inputTensors[3]->elementsSize()),
        },
        cpuOut(result->elementsSize()),
        out(result->elementsSize());
    std::iota(cpuIns[0].begin(), cpuIns[0].end(), 0);
    std::iota(cpuIns[1].begin(), cpuIns[1].end(), 0);
    std::iota(cpuIns[2].begin(), cpuIns[2].end(), 0);
    std::iota(cpuIns[3].begin(), cpuIns[3].end(), 0);
    gpuIns[0]->copyFromHost(cpuIns[0].data(), inputTensors[0]->bytesSize());
    gpuIns[1]->copyFromHost(cpuIns[1].data(), inputTensors[1]->bytesSize());
    gpuIns[2]->copyFromHost(cpuIns[2].data(), inputTensors[2]->bytesSize());
    gpuIns[3]->copyFromHost(cpuIns[3].data(), inputTensors[3]->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuIns[0], *gpuIns[1], *gpuIns[2], *gpuIns[3]};
        void *outputs[]{*gpuOut};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{cpuIns[0].data(), cpuIns[1].data(), cpuIns[2].data(), cpuIns[3].data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    gpuOut->copyToHost(out.data(), result->bytesSize());
    EXPECT_EQ(out, cpuOut);
}

#endif
