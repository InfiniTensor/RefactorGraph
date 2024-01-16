#ifdef USE_BANG

#include "../../../src/kernels/cast/cpu_kernel.hh"
#include "../../../src/kernels/cast/cnnl_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, CastCnnl) {
    // build routine
    auto x = Tensor::share(DataType::F32, Shape{2, 3, 4, 5});
    auto y = Tensor::share(DataType::I8, Shape{2, 3, 4, 5});
    auto kernel = CastCnnl::build(*x, *y),
         kCpu = CastCpu::build(*x, *y);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto xMlu = dev.malloc(x->bytesSize()),
         yMlu = dev.malloc(y->bytesSize());
    // put input data
    std::vector<float> x_(x->elementsSize());
    std::vector<int8_t> y_(y->elementsSize());
    std::iota(x_.begin(), x_.end(), 0);
    xMlu->copyFromHost(x_.data(), x->bytesSize());
    // inference
    {
        void const *inputs[]{*xMlu};
        void *outputs[]{*yMlu};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{x_.data()};
        void *outputs[]{y_.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<int8_t> result(y->elementsSize());
    yMlu->copyToHost(result.data(), y->bytesSize());
    EXPECT_EQ(result, y_);
}

#endif
