#ifdef USE_BANG

#include "../../../src/kernels/conv/cnnl_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

void testConvCnnl(int rank, const int64_t *pads, const int64_t *strides, const int64_t *dilations,
                  Shape xShape, Shape wShape, Shape yShape,
                  const std::vector<float> &ExpectData) {
    auto xTensor = Tensor::share(DataType::F32, xShape);
    auto wTensor = Tensor::share(DataType::F32, wShape);
    auto yTensor = Tensor::share(DataType::F32, yShape);
    PoolAttributes poolAttributes(rank, dilations, pads, strides);
    auto kernel = ConvCnnl::build(poolAttributes, *xTensor, *wTensor, std::nullopt, *yTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    // bang malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         xMlu = dev.malloc(xTensor->bytesSize()),
         wMlu = dev.malloc(wTensor->bytesSize()),
         yMlu = dev.malloc(yTensor->bytesSize());
    // put input data
    std::vector<int> xIncremental(xTensor->elementsSize()),
        wIncremental(wTensor->elementsSize());
    std::iota(xIncremental.begin(), xIncremental.end(), 0);
    std::iota(wIncremental.begin(), wIncremental.end(), 0);
    std::vector<float> xData(xIncremental.begin(), xIncremental.end()),
        wData(wIncremental.begin(), wIncremental.end());
    xMlu->copyFromHost(xData.data(), xTensor->bytesSize());
    wMlu->copyFromHost(wData.data(), wTensor->bytesSize());
    // inference
    void const *inputs[]{*xMlu, *wMlu};
    void *outputs[]{*yMlu};
    routine(res, *workspace, inputs, outputs);

    xMlu->copyToHost(xData.data(), xTensor->bytesSize());
    wMlu->copyToHost(wData.data(), wTensor->bytesSize());
    // fmt::println("{}", vec2str(xData));
    // fmt::println("{}", vec2str(wData));

    // std::vector<float> ws(workspaceSize);
    // workspace->copyToHost(ws.data(), workspaceSize);
    // fmt::println("{}", vec2str(ws));

    // take output data
    std::vector<float> result(yTensor->elementsSize());
    yMlu->copyToHost(result.data(), yTensor->bytesSize());
    // check
    for (auto i : range0_(ExpectData.size())) {
        EXPECT_FLOAT_EQ(ExpectData[i], result[i]);
    }
}

TEST(kernel, ConvCnnl) {
    int rank = 2;
    int64_t const
        pads[]{1, 1, 1, 1},
        strides[]{1, 1},
        dilations[]{1, 1};
    Shape
        xShape{1, 3, 3, 2},
        wShape{1, 3, 3, 2},
        yShape{1, 1, 3, 3};
    const std::vector<float> ExpectData = {570, 1158, 582, 888, 1785, 888, 582, 1158, 570};
    testConvCnnl(rank, pads, strides, dilations, xShape, wShape, yShape, ExpectData);
}


#endif
