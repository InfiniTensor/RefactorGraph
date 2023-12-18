#ifdef USE_CUDA

#include "../../../src/kernels/dynamic_quantize_linear/cpu_kernel.hh"
#include "../../../src/kernels/dynamic_quantize_linear/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, DynamicQuantizeLinearCuda) {
    auto size = 20;
    // build routine
    auto kernel = DynamicQuantizeLinearCuda::build(size),
         kCpu = DynamicQuantizeLinearCpu::build(size);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    auto rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto xGpu = dev.malloc(size * sizeof(float)),
         yGpu = dev.malloc(size * sizeof(uint8_t)),
         scaleGpu = dev.malloc(sizeof(float)),
         zpGpu = dev.malloc(sizeof(uint8_t)),
         workspace = dev.malloc(workspaceSize);
    // put input data
    std::vector<float> x(size);
    std::vector<uint8_t> y(size);
    float scale;
    uint8_t zeroPoint;
    for (auto i : range0_(size)) {
        x[i] = i * 3 + 15;
    }
    xGpu->copyFromHost(x.data());
    // inference
    {
        void const *inputs[]{*xGpu};
        void *outputs[]{*yGpu, *scaleGpu, *zpGpu};
        routine(res, *workspace, inputs, outputs);
    }
    {
        void const *inputs[]{x.data()};
        void *outputs[]{y.data(), &scale, &zeroPoint};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    {
        std::vector<uint8_t> result(size);
        yGpu->copyToHost(result.data());
        EXPECT_EQ(result, y);
    }
}

#endif
