#ifdef USE_CUDA

#include "../../../src/kernels/where/cpu_kernel.hh"
#include "../../../src/kernels/where/where_cuda.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, WhereCuda) {
    // build routine
    auto cTensor = Tensor::share(DataType::Bool, Shape{2, 5});
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 5});
    auto yTensor = Tensor::share(DataType::F32, Shape{3, 2, 5});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto kCpu = WhereCpu::build({*cTensor, *xTensor, *yTensor});
    auto kCuda = WhereCuda::build({*cTensor, *xTensor, *yTensor});
    ASSERT_TRUE(kCpu && kCuda);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res);
    auto rCuda = kCuda->lower(res);
    // malloc
    auto memManager = Target(Target::NvidiaGpu).memManager();
    auto gpuC = mem_manager::ForeignBlob::share(memManager, cTensor->bytesSize());
    auto gpuX = mem_manager::ForeignBlob::share(memManager, xTensor->bytesSize());
    auto gpuY = mem_manager::ForeignBlob::share(memManager, yTensor->bytesSize());
    auto gpuOut = mem_manager::ForeignBlob::share(memManager, outTensor->bytesSize());
    // put input data
    int dataC[cTensor->elementsSize()];
    memset(dataC, 1, cTensor->elementsSize() * sizeof(bool));
    gpuC->copyIn(dataC, cTensor->bytesSize());
    std::vector<float> dataX(xTensor->elementsSize());
    for (auto i : range0_(dataX.size())) { dataX[i] = 7; }
    gpuX->copyIn(dataX.data(), xTensor->bytesSize());
    std::vector<float> dataY(yTensor->elementsSize());
    for (auto i : range0_(dataY.size())) { dataY[i] = 3; }
    gpuY->copyIn(dataY.data(), yTensor->bytesSize());
    std::vector<float> cpuOut(outTensor->elementsSize());
    // inference
    {
        void const *inputs[]{dataC, dataX.data(), dataY.data()};
        void *outputs[]{cpuOut.data()};
        rCpu(res, inputs, outputs);
    }
    {
        void const *inputs[]{*gpuC, *gpuX, *gpuY};
        void *outputs[]{*gpuOut};
        rCuda(res, inputs, outputs);
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
