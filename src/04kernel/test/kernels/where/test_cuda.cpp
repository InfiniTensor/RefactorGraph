#ifdef USE_CUDA

#include "../../../src/kernels/where/cpu_kernel.hh"
#include "../../../src/kernels/where/where_cuda.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, WhereCuda) {
    // build routine
    auto cTensor = Tensor::share(DataType::Bool, Shape{2, 5});
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 5});
    auto yTensor = Tensor::share(DataType::F32, Shape{3, 2, 5});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto kCpu = WhereCpu::build({*cTensor, *xTensor, *yTensor});
    auto kCuda = WhereCuda::build({*cTensor, *xTensor, *yTensor});
    ASSERT_TRUE(kCpu && kCuda);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCuda = kCuda->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuC = dev.malloc(cTensor->bytesSize()),
         gpuX = dev.malloc(xTensor->bytesSize()),
         gpuY = dev.malloc(yTensor->bytesSize()),
         gpuOut = dev.malloc(outTensor->bytesSize());
    // put input data
    int dataC[cTensor->elementsSize()];
    memset(dataC, 1, cTensor->elementsSize() * sizeof(bool));
    gpuC->copyFromHost(dataC, cTensor->bytesSize());
    std::vector<float> dataX(xTensor->elementsSize());
    for (auto i : range0_(dataX.size())) { dataX[i] = 7; }
    gpuX->copyFromHost(dataX.data(), xTensor->bytesSize());
    std::vector<float> dataY(yTensor->elementsSize());
    for (auto i : range0_(dataY.size())) { dataY[i] = 3; }
    gpuY->copyFromHost(dataY.data(), yTensor->bytesSize());
    std::vector<float> cpuOut(outTensor->elementsSize());
    // inference
    {
        void const *inputs[]{dataC, dataX.data(), dataY.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuC, *gpuX, *gpuY};
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
