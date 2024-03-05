#ifdef USE_BANG

#include "../../../src/kernels/softmax/cnnl_kernel.hh"
#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, SoftmaxCnnl) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    dim_t axis = 2;
    auto kCpu = SoftmaxCpu::build(SoftmaxInfo(*xTensor, axis));
    auto kCnnl = SoftmaxCnnl::build(cnnl::SoftmaxAlgo::FAST, SoftmaxInfo(*xTensor, axis));
    ASSERT_TRUE(kCpu && kCnnl);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCnnl = kCnnl->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluIn = dev.malloc(xTensor->bytesSize()),
         mluOut = dev.malloc(outTensor->bytesSize());
    // put input data
    std::vector<float>
        data(xTensor->elementsSize(), 0),
        cpuOut(outTensor->elementsSize());
    mluIn->copyFromHost(data.data(), xTensor->bytesSize());
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*mluIn};
        void *outputs[]{*mluOut};
        rCnnl(res, nullptr, inputs, outputs);
        kernel::bang::sync();
    }
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    mluOut->copyToHost(result.data(), outTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
