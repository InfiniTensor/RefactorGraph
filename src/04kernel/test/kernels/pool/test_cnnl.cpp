#ifdef USE_BANG

#include "../../../src/kernels/pool/cnnl_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testPoolCnnl(PoolType poolType, int rank, const int64_t *pads, const int64_t *strides, KernelShape kernelShape, Shape xShape, Shape yShape, const std::vector<float> &ExpectData) {
    auto dataTensor = Tensor::share(DataType::F32, xShape);
    auto yTensor = Tensor::share(DataType::F32, yShape);
    //bool ceil = false;
    bool ceil = true;
    int64_t const dilations[] = {1, 1};
    PoolAttributes poolAttributes(rank, dilations, pads, strides);

    auto kernel = PoolCnnl::build(poolType, ceil, kernelShape, poolAttributes, *dataTensor, *yTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    // bang malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         mluMem = dev.malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 0.1f; }
    mluMem->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    void const *inputs[]{*mluMem};
    void *outputs[]{*mluMem};
    routine(res, *workspace, inputs, outputs);
    kernel::bang::sync();
    // take output data
    std::vector<float> result(yTensor->elementsSize());
    mluMem->copyToHost(result.data(), yTensor->bytesSize());
    // check
    for (auto i : range0_(ExpectData.size())) {
        EXPECT_FLOAT_EQ(ExpectData[i], result[i]);
    }
}

TEST(kernel, PoolCnnlMax) {
    int rank = 2;
    int64_t const
        pads[]{0, 0, 0, 0},
        strides[]{2, 2};
    KernelShape kernelShape{2, 2};
    Shape
        xShape{1, 1, 4, 4},
        yShape{1, 1, 2, 2};
    const std::vector<float> ExpectData = {0.5, 0.7, 1.3, 1.5};
    testPoolCnnl(PoolType::Max, rank, pads, strides, kernelShape, xShape, yShape, ExpectData);
}

TEST(kernel, PoolCnnlAvg) {
    int rank = 2;
    int64_t const
        pads[]{0, 0, 0, 0},
        strides[]{2, 2};
    KernelShape kernelShape{2, 2};
    Shape
        xShape{1, 1, 4, 4},
        yShape{1, 1, 2, 2};
    const std::vector<float> ExpectData = {0.25, 0.45, 1.05, 1.25};
    testPoolCnnl(PoolType::Average, rank, pads, strides, kernelShape, xShape, yShape, ExpectData);
}

#endif
