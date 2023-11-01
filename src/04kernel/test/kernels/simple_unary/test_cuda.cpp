﻿#ifdef USE_CUDA

#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include "../../../src/kernels/simple_unary/cuda_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

static void testOp(SimpleUnaryType opType) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = SimpleUnaryCuda::build(opType, *dataTensor);
    auto kCpu = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel && kCpu);
    auto routine = kernel->lower(),
         rCpu = kCpu->lower();
    // malloc
    auto gpuMem = mem_manager::ForeignBlob::share(
        Target(Target::NvidiaGpu).memManager(),
        dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
    gpuMem->copyIn(data.data(), dataTensor->bytesSize());
    // inference
    auto res = runtime::Resources();
    {
        void const *inputs[]{*gpuMem};
        void *outputs[]{*gpuMem};
        routine(res, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{data.data()};
        rCpu(res, inputs, outputs);
    }
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    gpuMem->copyOut(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(data[i], result[i]);
    }
}

TEST(kernel, SimpleUnaryCuda) {
    testOp(SimpleUnaryType::Abs);
    testOp(SimpleUnaryType::Relu);
    testOp(SimpleUnaryType::Sqrt);
    testOp(SimpleUnaryType::Sigmoid);
    testOp(SimpleUnaryType::Tanh);
}

#endif