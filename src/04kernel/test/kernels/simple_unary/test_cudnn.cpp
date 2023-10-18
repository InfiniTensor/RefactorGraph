#ifdef USE_CUDA

#include "../../../src/kernels/simple_unary/cudnn_activation_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, X) {
    // Build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 5});
    auto kernel = ActivationCudnn::build(SimpleUnaryType::Tanh, *dataTensor);
    ASSERT_TRUE(kernel);
    auto routine = kernel->lower();
    // cuda malloc
    auto memFn = Target(Target::NvidiaGpu).memFunc();
    auto gpuMem = mem_manager::ForeignBlob::share(memFn, dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (size_t i = 0; i < data.size(); i++) { data[i] = i * 0.1f; }
    memFn.copyHd(*gpuMem, data.data(), dataTensor->bytesSize());
    // inference
    auto res = runtime::Resources();
    void const *inputs[]{*gpuMem};
    void *outputs[]{*gpuMem};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    memFn.copyDh(result.data(), *gpuMem, dataTensor->bytesSize());
    // check
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(std::tanh(data[i]), result[i]);
    }
}

#endif// USE_CUDA
