#ifdef USE_CUDA

#include "../../../src/kernels/dequantize_linear/cpu_kernel.hh"
#include "../../../src/kernels/dequantize_linear/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, DequantizeLinearCuda) {
    // build routine
    auto x = Tensor::share(DataType::U8, {4});
    auto scale = Tensor::share(DataType::F32, {});
    auto zeroPoint = Tensor::share(DataType::U8, {});
    auto y = Tensor::share(DataType::F32, {4});
    auto kernel = DequantizeLinearCuda::build({*x, *scale, *zeroPoint}, *y),
         kCpu = DequantizeLinearCpu::build({*x, *scale, *zeroPoint}, *y);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    auto rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto xGpu = dev.malloc(x->bytesSize()),
         scaleGpu = dev.malloc(sizeof(float)),
         zpGpu = dev.malloc(sizeof(uint8_t)),
         yGpu = dev.malloc(y->bytesSize());
    // put input data
    std::vector<uint8_t> xData{0, 3, 128, 255};
    float scale_ = 2;
    uint8_t zp_ = 128;
    std::vector<float> yData(xData.size());
    xGpu->copyFromHost(xData.data());
    scaleGpu->copyFromHost(&scale_);
    zpGpu->copyFromHost(&zp_);
    // inference
    {
        void const *inputs[]{*xGpu, *scaleGpu, *zpGpu};
        void *outputs[]{*yGpu};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{xData.data(), &scale_, &zp_};
        void *outputs[]{yData.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    {
        std::vector<float> result(yData.size());
        yGpu->copyToHost(result.data());
        EXPECT_EQ(result, yData);
    }
}

#endif
