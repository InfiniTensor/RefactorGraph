#ifdef USE_CUDA
#include "../../../include/kernel/target.h"
#include "../../../src/kernels/reduce/cudnn_kernel.hh"
#include "runtime/mem_manager.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

static void testReducemean(const Shape &shape, const std::vector<float> &data,
                           Axes axes, const std::vector<float> ExpectData) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, shape);
    auto kernel = ReduceCudnn::build(axes, ReduceType::Mean, {*dataTensor});
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    res.fetchOrStore<runtime::MemManager>(Target(Target::NvidiaGpu).memManager());
    auto [routine, workspaceSize] = kernel->lower(res);
    // cuda malloc
    auto manager = Target(Target::NvidiaGpu).memManager();
    auto workspace = mem_manager::ForeignBlob::share(manager, workspaceSize);
    auto gpuMemIn = mem_manager::ForeignBlob::share(manager, dataTensor->bytesSize());
    auto gpuMemOut = mem_manager::ForeignBlob::share(manager, dataTensor->bytesSize());
    // put input output data
    gpuMemIn->copyIn(data.data(), dataTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*gpuMemIn};
        void *outputs[]{*gpuMemOut};
        routine(res, *workspace, inputs, outputs);
    }
    // take output data
    Shape outDimArray;
    std::unordered_set axesSet(axes.begin(), axes.end());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (axesSet.contains(i)) {
            outDimArray.push_back(shape[i]);
        }
    }
    auto outputTensor = Tensor::share(DataType::F32, outDimArray);
    std::vector<float> result(outDimArray.size());
    gpuMemOut->copyOut(result.data(), outputTensor->bytesSize());
    // check
    for (auto i : range0_(ExpectData.size())) {
        EXPECT_FLOAT_EQ(ExpectData[i], result[i]);
    }
}

TEST(kernel, ReduceMeanCudnn) {
    testReducemean({2, 3, 2, 2},
                   {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23},
                   {1, 2},
                   {5, 6, 17, 18});
    testReducemean({2, 3, 2, 2, 1},
                   {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23},
                   {1, 2},
                   {5, 6, 17, 18});
}

#endif
