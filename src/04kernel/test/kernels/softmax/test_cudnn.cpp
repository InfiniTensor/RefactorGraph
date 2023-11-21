#ifdef USE_CUDA

#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include "../../../src/kernels/softmax/cudnn_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, SoftmaxCudnn) {
    // build routine
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5, 4});
    dim_t axis = 2;
    auto kCpu = SoftmaxCpu::build(SoftmaxInfo(*xTensor, axis));
    auto kCudnn = SoftmaxCudnn::build(cudnn::SoftmaxAlgo::FAST, SoftmaxInfo(*xTensor, axis));
    ASSERT_TRUE(kCpu && kCudnn);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto rCudnn = kCudnn->lower(res).routine;
    // malloc
    auto memManager = Target(Target::NvidiaGpu).memManager();
    auto gpuX = mem_manager::ForeignBlob::share(memManager, xTensor->bytesSize());
    auto gpuOut = mem_manager::ForeignBlob::share(memManager, outTensor->bytesSize());
    // put input data
    std::vector<float> data(xTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 0.0; }
    gpuX->copyIn(data.data(), xTensor->bytesSize());
    std::vector<float> cpuOut(outTensor->elementsSize());
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuX};
        void *outputs[]{*gpuOut};
        rCudnn(res, nullptr, inputs, outputs);
    }
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    gpuOut->copyOut(result.data(), outTensor->bytesSize());
    // check
    for (auto i : range0_(result.size())) {
        EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
    }
}

#endif
