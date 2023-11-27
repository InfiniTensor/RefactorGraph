#ifdef USE_CUDA

#include "../../../src/kernels/pool/cudnn_kernel.hh"
#include "hardware/devices/nvidia.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testPoolCudnn(PoolType poolType, int rank, const int64_t *pads, const int64_t *strides, KernelShape kernelShape, Shape xShape, Shape yShape, const std::vector<float> &ExpectData) {
    auto dataTensor = Tensor::share(DataType::F32, xShape);
    auto yTensor = Tensor::share(DataType::F32, yShape);
    //bool ceil = false;
    bool ceil = true;
    int64_t const dilations[] = {1, 1};
    PoolAttributes poolAttributes(rank, dilations, pads, strides);

    auto kernel = PoolCudnn::build(poolType, ceil, kernelShape, poolAttributes, *dataTensor, *yTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // cuda malloc
    Device::register_<Nvidia>("nvidia");
    auto device = Device::init("nvidia", 0, "");
    auto gpuMem = device->malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 0.1f; }
    gpuMem->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    void const *inputs[]{*gpuMem};
    void *outputs[]{*gpuMem};
    routine(res, nullptr, inputs, outputs);
    // take output data
    std::vector<float> result(yTensor->elementsSize());
    gpuMem->copyToHost(result.data(), yTensor->bytesSize());
    // check
    for (auto i : range0_(ExpectData.size())) {
        EXPECT_FLOAT_EQ(ExpectData[i], result[i]);
    }
}

TEST(kernel, PoolCudnnMax) {
    int rank = 2;
    int64_t const dilations[] = {1, 1};
    int64_t const pads[] = {0, 0, 0, 0};
    int64_t const strides[] = {2, 2};
    KernelShape kernelShape = {2, 2};
    Shape xShape = {1, 1, 4, 4};
    Shape yShape = {1, 1, 2, 2};
    const std::vector<float> ExpectData = {0.5, 0.7, 1.3, 1.5};
    testPoolCudnn(PoolType::Max, rank, pads, strides, kernelShape, xShape, yShape, ExpectData);
}

TEST(kernel, PoolCudnnAvg) {
    int rank = 2;
    int64_t const dilations[] = {1, 1};
    int64_t const pads[] = {0, 0, 0, 0};
    int64_t const strides[] = {2, 2};
    KernelShape kernelShape = {2, 2};
    Shape xShape = {1, 1, 4, 4};
    Shape yShape = {1, 1, 2, 2};
    const std::vector<float> ExpectData = {0.25, 0.45, 1.05, 1.25};
    testPoolCudnn(PoolType::Average, rank, pads, strides, kernelShape, xShape, yShape, ExpectData);
}

#endif
