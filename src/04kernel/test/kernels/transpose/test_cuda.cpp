#ifdef USE_CUDA

#include "../../../src/kernels/transpose/cpu_kernel.hh"
#include "../../../src/kernels/transpose/cuda_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, TransposeCuda) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{1, 3, 2, 5});
    auto info = TransposeInfo(dataTensor->shape, Permutation{2, 3, 0, 1});
    auto kCpu = TransposeCpu::build(dataTensor->dataType, info);
    auto kernel = TransposeCuda::build(dataTensor->dataType, info);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res);
    auto routine = kernel->lower(res);
    // malloc
    auto memManager = Target(Target::NvidiaGpu).memManager();
    auto bytes = dataTensor->bytesSize();
    auto gpuIn = mem_manager::ForeignBlob::share(memManager, bytes),
         gpuOut = mem_manager::ForeignBlob::share(memManager, bytes);
    // put input data
    std::vector<float>
        cpuIn(dataTensor->elementsSize()),
        cpuOut(cpuIn.size());
    std::iota(cpuIn.begin(), cpuIn.end(), 0);
    gpuIn->copyIn(cpuIn.data(), bytes);
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
    gpuOut->copyOut(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
