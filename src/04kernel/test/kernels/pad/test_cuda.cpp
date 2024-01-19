#ifdef USE_CUDA

#include "../../../src/kernels/pad/cpu_kernel.hh"
#include "../../../src/kernels/pad/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, PadCuda) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 5});
    auto yTensor = Tensor::share(DataType::F32, Shape{4, 5, 5});
    PadsShape pads = {1, 1, 0, 1, 1, 0};
    PadType type = PadType::Constant;
    auto kernel = PadCuda::build(PadInfo(pads, type, *xTensor, *yTensor, false));
    auto kCpu = PadCpu::build(PadInfo(pads, type, *xTensor, *yTensor, false));
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuIn = dev.malloc(xTensor->bytesSize()),
         gpuOut = dev.malloc(yTensor->bytesSize());
    // put input data
    std::vector<float> data(xTensor->elementsSize()),
        cpuOut(yTensor->elementsSize());


    for (auto i : range0_(data.size())) { data[i] = i; }
    gpuIn->copyFromHost(data.data(), xTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOut};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(yTensor->elementsSize());
    gpuOut->copyToHost(result.data(), yTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
