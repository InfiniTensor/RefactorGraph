#ifdef USE_CUDA

#include "../../../src/kernels/rms_normalization/cpu_kernel.hh"
#include "../../../src/kernels/rms_normalization/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, RmsNormalizationCuda) {
    // build routine
    auto y = Tensor::share(DataType::F32, Shape{2, 3, 4});
    auto x = Tensor::share(DataType::F32, Shape{2, 3, 4});
    auto w = Tensor::share(DataType::F32, Shape{4});
    auto kernel = RmsNormalizationCuda::build(0, *x),
         kCpu = RmsNormalizationCpu::build(0, *x);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto yGpu = dev.malloc(y->bytesSize()),
         xGpu = dev.malloc(x->bytesSize()),
         wGpu = dev.malloc(w->bytesSize());
    // put input data
    std::vector<float> y_(y->elementsSize());
    std::vector<float> x_(x->elementsSize());
    std::vector<float> w_(w->elementsSize());
    std::iota(x_.begin(), x_.end(), 0);
    std::iota(w_.begin(), w_.end(), 1);
    xGpu->copyFromHost(x_.data(), x->bytesSize());
    wGpu->copyFromHost(w_.data(), w->bytesSize());
    // inference
    {
        void const *inputs[]{*xGpu, *wGpu};
        void *outputs[]{*yGpu};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{x_.data(), w_.data()};
        void *outputs[]{y_.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(y->elementsSize());
    yGpu->copyToHost(result.data(), y->bytesSize());
    for (auto i : range0_(y_.size())) {
        EXPECT_FLOAT_EQ(result[i], y_[i]);
    }
}

#endif
