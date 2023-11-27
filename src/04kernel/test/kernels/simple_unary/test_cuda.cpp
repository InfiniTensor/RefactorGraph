#ifdef USE_CUDA

#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include "../../../src/kernels/simple_unary/cuda_kernel.hh"
#include "hardware/devices/nvidia.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

static void testOp(SimpleUnaryType opType) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = SimpleUnaryCuda::build(opType, *dataTensor);
    auto kCpu = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    Device::register_<Nvidia>("nvidia");
    auto device = Device::init("nvidia", 0, "");
    auto gpuMem = device->malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
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

TEST(kernel, SimpleUnaryCuda) {
    testOp(SimpleUnaryType::Abs);
    testOp(SimpleUnaryType::Relu);
    testOp(SimpleUnaryType::Sqrt);
    testOp(SimpleUnaryType::Sigmoid);
    testOp(SimpleUnaryType::Tanh);
}

#endif
