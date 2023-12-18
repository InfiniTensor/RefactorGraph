#ifdef USE_CUDA

#include "../src/kernels/mat_mul_integer/cpu_kernel.hh"
#include "../src/kernels/mat_mul_integer/cublas_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, MatMulIntegerCublas) {
    // build routine
    auto A = Tensor::share(DataType::U8, Shape{1, 4});
    auto B = Tensor::share(DataType::U8, Shape{4, 12});
    auto Y = Tensor::share(DataType::I32, Shape{1, 12});
    MatMulIntegerInfo info(TensorRefs{*A, *B});
    auto cpuKernel = MatMulIntegerCpu::build(info);
    auto gpuKernel = MatMulIntegerCublas::build(info);
    ASSERT_TRUE(cpuKernel && gpuKernel);
    auto res = runtime::Resources();
    auto [cpuRoutine, workspace] = cpuKernel->lower(res);
    auto [gpuRoutine, workspace_] = gpuKernel->lower(res);
    ASSERT_EQ(workspace, workspace_);
    // put input data
    std::vector<uint8_t>
        dataA(A->elementsSize()),
        dataB(B->elementsSize());
    std::vector<int32_t>
        dataY(Y->elementsSize()),
        result(Y->elementsSize());
    std::iota(dataA.begin(), dataA.end(), 1);
    std::iota(dataB.data() + 0, dataB.data() + 12, 1);
    std::iota(dataB.data() + 12, dataB.data() + 24, 1);
    std::iota(dataB.data() + 24, dataB.data() + 36, 1);
    std::iota(dataB.data() + 36, dataB.data() + 48, 1);
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto ma = dev.malloc(A->bytesSize()),
         mb = dev.malloc(B->bytesSize()),
         my = dev.malloc(Y->bytesSize());
    ma->copyFromHost(dataA.data(), A->bytesSize());
    mb->copyFromHost(dataB.data(), B->bytesSize());
    // inference
    {
        void const *inputs[]{*ma, *mb};
        void *outputs[]{*my};
        gpuRoutine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{dataA.data(), dataB.data()};
        void *outputs[]{dataY.data()};
        cpuRoutine(res, nullptr, inputs, outputs);
    }
    // take output data
    my->copyToHost(result.data(), Y->bytesSize());
    // check
    EXPECT_EQ(result, dataY);
}

#endif
