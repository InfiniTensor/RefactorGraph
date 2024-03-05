#ifdef USE_BANG

#include "../../../src/kernels/slice/cnnl_kernel.hh"
#include "../../../src/kernels/slice/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, SliceCnnl) {
    // build routine
    Dimensions dims{
        {5, -2, 3},// 7 -> {5, 3, 1} -> {108, 900, -360}
        {2, 3, 2}, // 6 -> {2, 5}    -> { 36,  60,   90}
        {1, 1, 3}, // 5 -> {1, 2, 3} -> { 18,   6,   30}
        {0, 1, 1}, // 1 -> {0}
        {0, 1, 2}, // 2 -> {0, 1}
        {0, 1, 3}, // 3 -> {0, 1, 2}
    };
    auto input = Tensor::share(DataType::F32, Shape{7, 6, 5, 1, 2, 3}),
         output = Tensor::share(DataType::F32, Shape{3, 2, 3, 1, 2, 3});
    SliceInfo info(dims, *input);
    auto kernel = SliceCnnl::build(DataType::F32, dims, input->shape, output->shape);
    auto kCpu = SliceCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    auto rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluIn = dev.malloc(input->bytesSize()),
         mluOut = dev.malloc(output->bytesSize());
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        ans(output->elementsSize()),
        result(ans.size());
    std::iota(data.begin(), data.end(), 0);
    mluIn->copyFromHost(data.data(), input->bytesSize());
    // inference
    {
        void const *inputs[]{*mluIn};
        void *outputs[]{*mluOut};
        routine(res, nullptr, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{ans.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    mluOut->copyToHost(result.data(), output->bytesSize());
    EXPECT_EQ(result, ans);
}

#endif
