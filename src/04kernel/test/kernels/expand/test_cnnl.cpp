﻿#ifdef USE_BANG

#include "../../../src/kernels/expand/cnnl_kernel.hh"
#include "../../../src/kernels/expand/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ExpandCnnl) {
    // build routine
    auto input = Tensor::share(DataType::F32, {3, 4, 1, 6}),
         output = Tensor::share(DataType::F32, {2, 3, 4, 5, 6});
    auto kernel = ExpandCnnl::build(*input, *output);
    auto kCpu = ExpandCpu::build(ExpandInfo(*input, *output));
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
