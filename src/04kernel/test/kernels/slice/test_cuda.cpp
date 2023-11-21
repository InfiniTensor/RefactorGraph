#ifdef USE_CUDA

#include "../../../src/kernels/slice/cpu_kernel.hh"
#include "../../../src/kernels/slice/cuda_kernel.hh"
#include "kernel/target.h"
#include "runtime/mem_manager.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SliceCuda) {
    // build routine
    Dimensions dims{
        {5, -2, 3},// 7 -> {5, 3, 1} -> {108, 900, -360}
        {2, 3, 2}, // 6 -> {2, 5}    -> { 36,  60,   90}
        {1, 1, 3}, // 5 -> {1, 2, 3} -> { 18,   6,   30}
        {0, 1, 1}, // 1 -> {0}
        {0, 1, 2}, // 2 -> {0, 1}
        {0, 1, 3}, // 3 -> {0, 1, 2}
    };
    auto input = Tensor::share(DataType::F32, Shape{7, 6, 5, 1, 2, 3}),
         output = Tensor::share(DataType::F32, Shape{3, 2, 3, 1, 2, 3});
    SliceInfo info(dims, *input);
    auto kernel = SliceCuda::build(info);
    auto kCpu = SliceCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    auto rCpu = kCpu->lower(res).routine;
    // malloc
    auto memManager = Target(Target::NvidiaGpu).memManager();
    Arc<mem_manager::ForeignBlob>
        gpuIn = mem_manager::ForeignBlob::share(memManager, input->bytesSize()),
        gpuOut = mem_manager::ForeignBlob::share(memManager, output->bytesSize());
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        ans(output->elementsSize()),
        result(ans.size());
    std::iota(data.begin(), data.end(), 0);
    gpuIn->copyIn(data.data(), input->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuOut};
        routine(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{ans.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    gpuOut->copyOut(result.data(), output->bytesSize());
    EXPECT_EQ(result, ans);
}

#endif
