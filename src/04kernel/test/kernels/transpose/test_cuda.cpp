#ifdef USE_CUDA

#include "../../../src/kernels/transpose/cpu_kernel.hh"
#include "../../../src/kernels/transpose/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, TransposeCuda) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{1, 3, 2, 5});
    auto info = TransposeInfo(dataTensor->dataType, dataTensor->shape, Permutation{2, 3, 0, 1});
    auto kCpu = TransposeCpu::build(info);
    auto kernel = TransposeCuda::build(info);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto routine = kernel->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto bytes = dataTensor->bytesSize();
    auto gpuIn = dev.malloc(bytes),
         gpuOut = dev.malloc(bytes);
    // put input data
    std::vector<float>
        cpuIn(dataTensor->elementsSize()),
        cpuOut(cpuIn.size());
    std::iota(cpuIn.begin(), cpuIn.end(), 0);
    gpuIn->copyFromHost(cpuIn.data(), bytes);
    // inference
    {
        void const *inputs[]{cpuIn.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOut};
        routine(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    gpuOut->copyToHost(result.data(), bytes);
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
