#ifdef USE_BANG

#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include "../../../src/kernels/simple_unary/cnnl_activation_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ActivationCnnl) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kCpu = SimpleUnaryCpu::build(SimpleUnaryType::Tanh, *dataTensor);
    ASSERT_TRUE(kCpu);
    auto kernel = ActivationCnnl::build(SimpleUnaryType::Tanh, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto routine = kernel->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto gpuMem = dev.malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
    gpuMem->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{data.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuMem};
        void *outputs[]{*gpuMem};
        routine(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(data.size());
    gpuMem->copyToHost(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_NEAR(data[i], result[i], 1e-4);
    }
}

#endif// USE_BANG
