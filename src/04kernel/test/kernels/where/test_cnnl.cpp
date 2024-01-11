#ifdef USE_BANG

#include "../../../src/kernels/where/cnnl_kernel.hh"
#include "../../../src/kernels/where/cpu_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testWhereCnnl(Shape cDim, Shape xDim, Shape yDim, Shape outDim) {
    // build routine
    auto cTensor = Tensor::share(DataType::Bool, cDim);
    auto xTensor = Tensor::share(DataType::F32, xDim);
    auto yTensor = Tensor::share(DataType::F32, yDim);
    auto outTensor = Tensor::share(DataType::F32, outDim);
    auto kCpu = WhereCpu::build({*cTensor, *xTensor, *yTensor});
    auto kCnnl = WhereCnnl::build({*cTensor, *xTensor, *yTensor}, {*outTensor});
    ASSERT_TRUE(kCpu && kCnnl);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto [rCnnl, workspaceSize] = kCnnl->lower(res);
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         mluC = dev.malloc(cTensor->bytesSize()),
         mluX = dev.malloc(xTensor->bytesSize()),
         mluY = dev.malloc(yTensor->bytesSize()),
         mluOut = dev.malloc(outTensor->bytesSize());
    // put input data
    int dataC[cTensor->elementsSize()];
    memset(dataC, 1, cTensor->elementsSize() * sizeof(bool));
    mluC->copyFromHost(dataC, cTensor->bytesSize());
    std::vector<float> dataX(xTensor->elementsSize());
    for (auto i : range0_(dataX.size())) { dataX[i] = 7; }
    mluX->copyFromHost(dataX.data(), xTensor->bytesSize());
    std::vector<float> dataY(yTensor->elementsSize());
    for (auto i : range0_(dataY.size())) { dataY[i] = 3; }
    mluY->copyFromHost(dataY.data(), yTensor->bytesSize());
    std::vector<float> cpuOut(outTensor->elementsSize());
    // inference
    {
        void const *inputs[]{dataC, dataX.data(), dataY.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*mluC, *mluX, *mluY};
        void *outputs[]{*mluOut};
        rCnnl(res, *workspace, inputs, outputs);
    }
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    mluOut->copyToHost(result.data(), outTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

TEST(kernel, WhereCnnl) {
    testWhereCnnl(Shape{2, 5}, Shape{2, 3, 1, 5}, Shape{3, 2, 5}, Shape{2, 3, 2, 5});
    testWhereCnnl(Shape{1}, Shape{4}, Shape{1}, Shape{4});
    testWhereCnnl(Shape{3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3});
}

#endif
