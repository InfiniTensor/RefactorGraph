#ifdef USE_CUDA

#include "../../../src/kernels/simple_unary/cudnn_activation_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, ActivationCudnn) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = ActivationCudnn::build(SimpleUnaryType::Tanh, *dataTensor);
    ASSERT_TRUE(kernel);
    auto routine = kernel->lower();
    // cuda malloc
    auto gpuMem = mem_manager::ForeignBlob::share(
        Target(Target::NvidiaGpu).memFunc(),
        dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 0.1f; }
    gpuMem->copyIn(data.data(), dataTensor->bytesSize());
    // inference
    auto res = runtime::Resources();
    void const *inputs[]{*gpuMem};
    void *outputs[]{*gpuMem};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    gpuMem->copyOut(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(std::tanh(data[i]), result[i]);
    }
}

#endif// USE_CUDA
