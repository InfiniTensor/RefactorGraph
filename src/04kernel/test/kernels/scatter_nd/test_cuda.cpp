#ifdef USE_CUDA

#include "../../../src/kernels/scatter_nd/cpu_kernel.hh"
#include "../../../src/kernels/scatter_nd/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ScatterNDCuda) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{8});
    auto indices = Tensor::share(DataType::I64, Shape{4, 1});
    auto updates = Tensor::share(DataType::F32, Shape{4});
    ScatterNDInfo info(*data, *indices);
    auto kernel = ScatterNDCuda::build(info),
         kCpu = ScatterNDCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuData = dev.malloc(data->bytesSize()),
         gpuIndices = dev.malloc(indices->bytesSize()),
         gpuUpdates = dev.malloc(updates->bytesSize());
    // put input data
    std::vector<float> data_(data->elementsSize());
    std::iota(data_.begin(), data_.end(), 1);
    std::vector<int64_t> indices_{4, 3, 1, 7};
    std::vector<float> updates_{9, 10, 11, 12};
    gpuData->copyFromHost(data_.data(), data->bytesSize());
    gpuIndices->copyFromHost(indices_.data(), indices->bytesSize());
    gpuUpdates->copyFromHost(updates_.data(), updates->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuData, *gpuIndices, *gpuUpdates};
        void *outputs[]{*gpuData};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data_.data(), indices_.data(), updates_.data()};
        void *outputs[]{data_.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(data->elementsSize());
    gpuData->copyToHost(result.data(), data->bytesSize());
    EXPECT_EQ(result, data_);
}

#endif
