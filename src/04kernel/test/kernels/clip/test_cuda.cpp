#ifdef USE_CUDA

#include "../../../src/kernels/clip/cpu_kernel.hh"
#include "../../../src/kernels/clip/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ClipCuda) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{2, 3, 4, 5});
    auto kernel = ClipCuda::build(*data, true),
         kCpu = ClipCpu::build(*data, true);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuMem = dev.malloc(data->bytesSize()),
         gpuMin = dev.malloc(sizeof(float)),
         gpuMax = dev.malloc(sizeof(float));
    // put input data
    std::vector<float> value(data->elementsSize());
    float min = 30, max = 80;
    std::iota(value.begin(), value.end(), 0);
    gpuMem->copyFromHost(value.data(), data->bytesSize());
    gpuMin->copyFromHost(&min, sizeof(float));
    gpuMax->copyFromHost(&max, sizeof(float));
    // inference
    {
        void const *inputs[]{*gpuMem, *gpuMin, *gpuMax};
        void *outputs[]{*gpuMem};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{value.data(), &min, &max};
        void *outputs[]{value.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(data->elementsSize());
    gpuMem->copyToHost(result.data(), data->bytesSize());
    EXPECT_EQ(result, value);
}

#endif
