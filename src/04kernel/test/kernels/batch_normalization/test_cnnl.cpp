#ifdef USE_BANG

#include "../../../src/kernels/batch_normalization/cnnl_kernel.hh"
#include "../../../src/kernels/batch_normalization/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, BatchNormalizationCnnl) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{1, 2, 3, 2});
    auto outTensor = Tensor::share(DataType::F32, Shape{1, 2, 3, 2});
    auto scaleTensor = Tensor::share(DataType::F32, Shape{2});
    auto biasTensor = Tensor::share(DataType::F32, Shape{2});
    auto meanTensor = Tensor::share(DataType::F32, Shape{2});
    auto varTensor = Tensor::share(DataType::F32, Shape{2});
    float epsilon = 0.00001;
    TensorRefs inputs = TensorRefs{*xTensor, *scaleTensor, *biasTensor, *meanTensor, *varTensor};
    auto kCpu = BatchNormalization::build(epsilon, inputs);
    auto kCnnl = BatchNormalizationCnnl::build(epsilon, inputs);
    ASSERT_TRUE(kCpu && kCnnl);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto [rMlu, workspaceSize] = kCnnl->lower(res);
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         mluIn = dev.malloc(xTensor->bytesSize()),
         mluScale = dev.malloc(scaleTensor->bytesSize()),
         mluBias = dev.malloc(biasTensor->bytesSize()),
         mluMean = dev.malloc(meanTensor->bytesSize()),
         mluVar = dev.malloc(varTensor->bytesSize()),
         mluOut = dev.malloc(outTensor->bytesSize());
    // put input data
    std::vector<float>
        data(xTensor->elementsSize(), 1.0f),
        scale(scaleTensor->elementsSize(), 0.5f),
        bias(biasTensor->elementsSize(), 1.0f),
        mean(meanTensor->elementsSize(), 0.5f),
        var(varTensor->elementsSize(), 1.0f),
        cpuOut(outTensor->elementsSize());
    mluIn->copyFromHost(data.data(), xTensor->bytesSize());
    mluScale->copyFromHost(scale.data(), scaleTensor->bytesSize());
    mluBias->copyFromHost(bias.data(), biasTensor->bytesSize());
    mluMean->copyFromHost(mean.data(), meanTensor->bytesSize());
    mluVar->copyFromHost(var.data(), varTensor->bytesSize());
    // inference
    {
        void const *inputs[]{data.data(), scale.data(), bias.data(), mean.data(), var.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*mluIn, *mluScale, *mluBias, *mluMean, *mluVar};
        void *outputs[]{*mluOut};
        rMlu(res, *workspace, inputs, outputs);
        kernel::bang::sync();
    }
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    mluOut->copyToHost(result.data(), outTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
