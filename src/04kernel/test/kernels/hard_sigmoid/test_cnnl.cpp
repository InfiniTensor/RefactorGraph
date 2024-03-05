#ifdef USE_BANG

#include "../../../src/kernels/hard_sigmoid/cnnl_kernel.hh"
#include "../../../src/kernels/hard_sigmoid/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, HardSigmoidCnnl) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 5});
    float alpha = 0.2f, beta = 0.5f;
    auto kernel = HardSigmoidCnnl::build(alpha, beta, *dataTensor);
    auto kCpu = HardSigmoidCpu::build(alpha, beta, *dataTensor);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluMem = dev.malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i; }
    mluMem->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*mluMem};
        void *outputs[]{*mluMem};
        routine(res, nullptr, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{data.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    mluMem->copyToHost(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(data[i], result[i]);
    }
}

#endif
