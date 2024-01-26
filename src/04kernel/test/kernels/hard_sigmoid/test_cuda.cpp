#ifdef USE_CUDA

#include "../../../src/kernels/hard_sigmoid/cpu_kernel.hh"
#include "../../../src/kernels/hard_sigmoid/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, HardSigmoidCuda) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 5});
    float alpha = 0.2f, beta = 0.5f;
    auto kernel = HardSigmoidCuda::build(alpha, beta, *dataTensor);
    auto kCpu = HardSigmoidCpu::build(alpha, beta, *dataTensor);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuMem = dev.malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i; }
    gpuMem->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuMem};
        void *outputs[]{*gpuMem};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{data.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    gpuMem->copyToHost(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(data[i], result[i]);
    }
}

#endif
