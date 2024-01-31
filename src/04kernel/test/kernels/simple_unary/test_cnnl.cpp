#ifdef USE_BANG

#include "../../../src/kernels/simple_unary/cnnl_activation_kernel.hh"
#include "../../../src/kernels/simple_unary/cnnl_simple_unary_kernel.hh"
#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

static void testOp(SimpleUnaryType opType, bool activation = true) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = activation ? ActivationCnnl::build(opType, *dataTensor)
                             : SimpleUnaryCnnl::build(opType, *dataTensor);
    auto kCpu = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluMem = dev.malloc(dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
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
        EXPECT_NEAR(data[i], result[i], 1e-4);
    }
}

TEST(kernel, SimpleUnaryCnnl) {
    testOp(SimpleUnaryType::Abs, false);
    testOp(SimpleUnaryType::Neg, false);
    testOp(SimpleUnaryType::Sqrt, false);
    testOp(SimpleUnaryType::Erf, false);
}

TEST(kernel, ActivationCnnl) {
    testOp(SimpleUnaryType::Relu);
    testOp(SimpleUnaryType::Sigmoid);
    testOp(SimpleUnaryType::Tanh);
    testOp(SimpleUnaryType::HardSwish);
}


#endif// USE_BANG
