#ifdef USE_CUDA

#include "../../../src/kernels/expand/cpu_kernel.hh"
#include "../../../src/kernels/expand/cuda_kernel.hh"
#include "kernel/target.h"
#include "runtime/mem_manager.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, ExpandCuda) {
    // build routine
    auto input = Tensor::share(DataType::F32, {3, 4, 1, 6}),
         output = Tensor::share(DataType::F32, {2, 3, 4, 5, 6});
    ExpandInfo info(*input, *output);
    auto kernel = ExpandCuda::build(info);
    auto kCpu = ExpandCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    auto rCpu = kCpu->lower(res);
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
        routine(res, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{ans.data()};
        rCpu(res, inputs, outputs);
    }
    // check
    gpuOut->copyOut(result.data(), output->bytesSize());
    EXPECT_EQ(result, ans);
}

#endif
