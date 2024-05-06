#ifdef USE_CUDA

#include "../../../src/kernels/topk/cpu_kernel.hh"
#include "../../../src/kernels/topk/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, TopKCuda) {
    // build routine 
    auto inputTensor = Tensor::share(DataType::F32, Shape{3, 4});
    std::vector<Arc<Tensor>> outputTensors{
            Tensor::share(DataType::F32, Shape{3, 3}),
            Tensor::share(DataType::U32, Shape{3, 3})};

    auto kCpu = TopKCpu::build(TopKInfo(3,1, *inputTensor));
    auto kCuda = TopKCuda::build(TopKInfo(3,1, *inputTensor));
    ASSERT_TRUE(kCpu);
    ASSERT_TRUE(kCuda);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCuda = kCuda->lower(res).routine;

    // device malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    Arc<Device::Blob>
        gpuIn = dev.malloc(inputTensor->bytesSize()),
        gpuOuts[]{
            dev.malloc(outputTensors[0]->bytesSize()),
            dev.malloc(outputTensors[1]->bytesSize()),
        };
    // put input data
    std::vector<float>  data(inputTensor->elementsSize());

    std::vector<float>    outCpu1(outputTensors[0]->elementsSize());
    std::vector<uint32_t> outCpu2(outputTensors[1]->elementsSize());

        
    std::vector<float> out1(outputTensors[0]->elementsSize());
    std::vector<uint32_t> out2(outputTensors[1]->elementsSize());

    std::iota(data.begin(), data.end(), 0);
    gpuIn->copyFromHost(data.data(), inputTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOuts[0], *gpuOuts[1]};
        rCuda(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{outCpu1.data(), outCpu2.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    
    gpuOuts[0]->copyToHost(out1.data(), outputTensors[0]->bytesSize());
    EXPECT_EQ(out1, outCpu1);
    gpuOuts[1]->copyToHost(out2.data(), outputTensors[1]->bytesSize());
    EXPECT_EQ(out2, outCpu2);
    
}

#endif
