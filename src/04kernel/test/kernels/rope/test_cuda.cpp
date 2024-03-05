#ifdef USE_CUDA

#include "../../../src/kernels/rope/cpu_kernel.hh"
#include "../../../src/kernels/rope/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, RoPECuda) {
    auto input = Tensor::share(DataType::F32, Shape{2, 3, 1, 2});
    auto posIDs = Tensor::share(DataType::I64, Shape{2, 3});
    auto output = Tensor::share(DataType::F32, Shape{2, 3, 1, 2});
    auto res = runtime::Resources();
    RoPEInfo info(*input, 10000.0f);
    auto cudaKernel = RoPECuda::build(info, *input)->lower(res).routine;
    auto cpuKernel = RoPECpu::build(info, *input)->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto inputGpu = dev.malloc(input->bytesSize()),
         posIDsGpu = dev.malloc(posIDs->bytesSize()),
         outputGpu = dev.malloc(output->bytesSize());
    // put input data
    std::vector<float> output_(output->elementsSize());
    std::vector<int64_t> posIDs_({0, 1, 2, 0, 1, 2});
    std::vector<float> input_(input->elementsSize(), 0.2);

    inputGpu->copyFromHost(input_.data(), input->bytesSize());
    posIDsGpu->copyFromHost(posIDs_.data(), posIDs->bytesSize());
    {
        void const *inputs[]{*inputGpu, *posIDsGpu};
        void *outputs[]{*outputGpu};
        cudaKernel(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{input_.data(), posIDs_.data()};
        void *outputs[]{output_.data()};
        cpuKernel(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(output->elementsSize());
    outputGpu->copyToHost(result.data(), output->bytesSize());
    for (auto i : range0_(output_.size())) {
        EXPECT_FLOAT_EQ(result[i], output_[i]);
    }
}

#endif
