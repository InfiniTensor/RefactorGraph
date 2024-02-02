#ifdef USE_CUDA

#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include "../../../src/kernels/softmax/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

static void test(Shape shape, int axis) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, shape);
    auto outTensor = Tensor::share(DataType::F32, shape);
    SoftmaxInfo info(*xTensor, axis);
    auto kCpu = SoftmaxCpu::build(info);
    auto kCuda = SoftmaxCuda::build(info);
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
    std::iota(data.begin(), data.end(), 0);
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

TEST(kernel, SoftmaxCuda) {
    test({2, 3, 2, 5, 4}, 1);
    test({2, 2048, 2, 5, 4}, 1);
}

#endif
