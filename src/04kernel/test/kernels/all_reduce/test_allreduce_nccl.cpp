#ifdef USE_CUDA

#include "../src/kernels/all_reduce/nccl_kernel.hh"
#include "../src/utilities/cuda/nccl_communicator.hh"
#include "kernel/cuda/functions.cuh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <thread>

using namespace refactor;
using namespace kernel;
using namespace nccl;
using namespace cuda;
using namespace hardware;

void allReduce(AllReduceType redType, int rank, int worldSize, std::vector<float> data, std::vector<float> ans) {
    cuda::setCudaDevice(rank);
    auto &dev = *device::init(Device::Type::Nvidia, rank, "");
    auto input = Tensor::share(DataType::F32, Shape{2}, LayoutType::NCHW);
    auto output = Tensor::share(DataType::F32, Shape{2}, LayoutType::NCHW);
    auto kernel = AllReduceNccl::build(redType, *input, *output);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    res.fetchOrStore<NcclCommunicator>(worldSize, rank);
    auto routine = kernel->lower(res).routine;

    
    auto inGPU = dev.malloc(input->bytesSize());
    auto outGPU = dev.malloc(output->bytesSize());
    inGPU->copyFromHost(data.data(), input->bytesSize());

    void const *inputs[]{*inGPU};
    void *outputs[]{*outGPU};
    routine(res, nullptr, inputs, outputs);

    std::vector<float> result(output->elementsSize());
    outGPU->copyToHost(result.data(), output->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(ans[i], result[i]);
    }
}


TEST(kernel, NCCL_AllReduceSum) {
    std::vector<float> data[2] = {{2., 3.}, {5., 6.}};
    std::vector<float> ans = {7., 9.};
    int worldSize = 2;

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < worldSize; ++gpu) {
        threads.emplace_back(allReduce, AllReduceType::Sum,
                             gpu, worldSize, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    // Reset device context for following tests
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    dev.setContext();
}
#endif
