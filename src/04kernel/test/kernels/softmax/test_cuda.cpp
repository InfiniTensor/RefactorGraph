#ifdef USE_CUDA

#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include "../../../src/kernels/softmax/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, SoftmaxCuda) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    dim_t axis = 1;
    auto kCpu = SoftmaxCpu::build(SoftmaxInfo(*xTensor, axis));
    auto kCuda = SoftmaxCuda::build(SoftmaxInfo(*xTensor, axis));
    ASSERT_TRUE(kCpu && kCuda);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCuda = kCuda->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuIn = dev.malloc(xTensor->bytesSize()),
         gpuOut = dev.malloc(outTensor->bytesSize());
    // put input data
    std::vector<float>
        data(xTensor->elementsSize(), 0),
        cpuOut(outTensor->elementsSize());
    gpuIn->copyFromHost(data.data(), xTensor->bytesSize());
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOut};
        rCuda(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    gpuOut->copyToHost(result.data(), outTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
