#ifdef USE_CUDA

#include "../../../src/kernels/split/cpu_kernel.hh"
#include "../../../src/kernels/split/cuda_kernel.hh"
#include "kernel/target.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SplitCuda) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 20, 7, 7});
    std::vector<Arc<Tensor>> outputTensors{
        Tensor::share(DataType::F32, Shape{2, 3, 1, 1, 7, 7}),// 勿
        Tensor::share(DataType::F32, Shape{2, 3, 1, 9, 7, 7}),// 忘
        Tensor::share(DataType::F32, Shape{2, 3, 1, 3, 7, 7}),// 国
        Tensor::share(DataType::F32, Shape{2, 3, 1, 7, 7, 7}),// 耻
    };
    TensorRefs outputs_;
    outputs_.reserve(outputTensors.size());
    std::transform(outputTensors.begin(), outputTensors.end(),
                   std::back_inserter(outputs_),
                   [](auto const &it) { return std::cref(*it); });
    auto info = SplitInfo(3, outputs_);
    auto kCpu = SplitCpu::build(info);
    auto kernel = SplitCuda::build(info);
    ASSERT_TRUE(kCpu && kernel);
    auto rCpu = kCpu->lower();
    auto routine = kernel->lower();
    // malloc
    auto memManager = Target(Target::NvidiaGpu).memManager();
    Arc<mem_manager::ForeignBlob>
        gpuIn = mem_manager::ForeignBlob::share(memManager, dataTensor->bytesSize()),
        gpuOuts[]{
            mem_manager::ForeignBlob::share(memManager, outputTensors[0]->bytesSize()),
            mem_manager::ForeignBlob::share(memManager, outputTensors[1]->bytesSize()),
            mem_manager::ForeignBlob::share(memManager, outputTensors[2]->bytesSize()),
            mem_manager::ForeignBlob::share(memManager, outputTensors[3]->bytesSize()),
        };
    // put input data
    std::vector<float>
        data(dataTensor->elementsSize()),
        outsCpu[]{
            std::vector<float>(outputTensors[0]->elementsSize()),
            std::vector<float>(outputTensors[1]->elementsSize()),
            std::vector<float>(outputTensors[2]->elementsSize()),
            std::vector<float>(outputTensors[3]->elementsSize()),
        },
        outs[]{
            std::vector<float>(outputTensors[0]->elementsSize()),
            std::vector<float>(outputTensors[1]->elementsSize()),
            std::vector<float>(outputTensors[2]->elementsSize()),
            std::vector<float>(outputTensors[3]->elementsSize()),
        };
    std::iota(data.begin(), data.end(), 0);
    gpuIn->copyIn(data.data(), dataTensor->bytesSize());
    // inference
    auto res = runtime::Resources();
    {
        void const *inputs[]{*gpuIn};
        void *outputs[]{*gpuIn, *gpuIn, *gpuIn, *gpuIn};
        routine(res, inputs, outputs);
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{outsCpu[0].data(), outsCpu[1].data(), outsCpu[2].data(), outsCpu[3].data()};
        rCpu(res, inputs, outputs);
    }
    // check
    for (auto i : range0_(outputTensors.size())) {
        gpuOuts[i]->copyOut(outs[i].data(), outputTensors[i]->bytesSize());
        // EXPECT_EQ(outs[i], outsCpu[i]);
    }
}

#endif
