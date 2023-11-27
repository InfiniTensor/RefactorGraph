#ifdef USE_CUDA

#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include "../../../src/kernels/softmax/cudnn_kernel.hh"
#include "hardware/devices/nvidia.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, SoftmaxCudnn) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    dim_t axis = 2;
    auto kCpu = SoftmaxCpu::build(SoftmaxInfo(*xTensor, axis));
    auto kCudnn = SoftmaxCudnn::build(cudnn::SoftmaxAlgo::FAST, SoftmaxInfo(*xTensor, axis));
    ASSERT_TRUE(kCpu && kCudnn);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCudnn = kCudnn->lower(res).routine;
    // malloc
    Device::register_<Nvidia>("nvidia");
    auto device = Device::init("nvidia", 0, "");
    auto gpuIn = device->malloc(xTensor->bytesSize()),
         gpuOut = device->malloc(outTensor->bytesSize());
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
        rCudnn(res, nullptr, inputs, outputs);
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
