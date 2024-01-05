#ifdef USE_BANG

#include "../../../src/kernels/transpose/cnnl_kernel.hh"
#include "../../../src/kernels/transpose/cpu_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, TransposeCnnl) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{1, 3, 2, 5});
    auto info = TransposeInfo(dataTensor->shape, Permutation{2, 3, 0, 1});
    auto kCpu = TransposeCpu::build(dataTensor->dataType, info);
    auto kernel = TransposeCnnl::build(dataTensor->dataType, dataTensor->shape, Permutation{2, 3, 0, 1});
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto [routine, workspaceSize] = kernel->lower(res);
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto bytes = dataTensor->bytesSize();
    auto workspace = dev.malloc(workspaceSize),
         mluIn = dev.malloc(bytes),
         mluOut = dev.malloc(bytes);
    // put input data
    std::vector<float>
        cpuIn(dataTensor->elementsSize()),
        cpuOut(cpuIn.size());
    std::iota(cpuIn.begin(), cpuIn.end(), 0);
    mluIn->copyFromHost(cpuIn.data(), bytes);
    // inference
    {
        void const *inputs[]{cpuIn.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*mluIn};
        void *outputs[]{*mluOut};
        routine(res, *workspace, inputs, outputs);
    }
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    mluOut->copyToHost(result.data(), bytes);
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
