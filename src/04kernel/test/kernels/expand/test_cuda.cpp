#ifdef USE_CUDA

#include "../../../src/kernels/expand/cpu_kernel.hh"
#include "../../../src/kernels/expand/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ExpandCuda) {
    // build routine
    auto input = Tensor::share(DataType::F32, {3, 4, 1, 6}),
         output = Tensor::share(DataType::F32, {2, 3, 4, 5, 6});
    ExpandInfo info(*input, *output);
    auto kernel = ExpandCuda::build(info);
    auto kCpu = ExpandCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    auto rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto gpuIn = dev.malloc(input->bytesSize()),
         gpuOut = dev.malloc(output->bytesSize());
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        ans(output->elementsSize()),
        result(ans.size());
    std::iota(data.begin(), data.end(), 0);
    gpuIn->copyFromHost(data.data(), input->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOut};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{ans.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    gpuOut->copyToHost(result.data(), output->bytesSize());
    EXPECT_EQ(result, ans);
}

#endif
