#ifdef USE_BANG

#include "../../../src/kernels/clip/cnnl_kernel.hh"
#include "../../../src/kernels/clip/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ClipCnnl) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{2, 3, 4, 5});
    auto kernel = ClipCnnl::build(*data, true),
         kCpu = ClipCpu::build(*data, true);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluMem = dev.malloc(data->bytesSize()),
         mluMin = dev.malloc(sizeof(float)),
         mluMax = dev.malloc(sizeof(float));
    // put input data
    std::vector<float> value(data->elementsSize());
    float min = 30, max = 80;
    std::iota(value.begin(), value.end(), 0);
    mluMem->copyFromHost(value.data(), data->bytesSize());
    mluMin->copyFromHost(&min, sizeof(float));
    mluMax->copyFromHost(&max, sizeof(float));
    // inference
    {
        void const *inputs[]{*mluMem, *mluMin, *mluMax};
        void *outputs[]{*mluMem};
        routine(res, nullptr, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{value.data(), &min, &max};
        void *outputs[]{value.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(data->elementsSize());
    mluMem->copyToHost(result.data(), data->bytesSize());
    EXPECT_EQ(result, value);
}

#endif
