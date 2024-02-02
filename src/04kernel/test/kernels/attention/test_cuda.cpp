#ifdef USE_CUDA

#include "../../../src/kernels/attention/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, AttentionCudaNoKvCache) {
    // build routine
    AttentionInfo info{
        .dataType = DataType::FP16,
        .batch = 1,
        .nHead = 4,
        .nKVHead = 4,
        .seqLen = 31,
        .headDim = 256,
        .cacheLen = 0,
        .concatCache = false,
        .resetCache = false,
    };
    auto q = Tensor::share(DataType::FP16, Shape{info.batch, info.nHead, info.seqLen, info.headDim}),
         k = Tensor::share(DataType::FP16, Shape{info.batch, info.nKVHead, info.seqLen, info.headDim}),
         v = Tensor::share(DataType::FP16, Shape{info.batch, info.nKVHead, info.seqLen, info.headDim}),
         o = q;
    auto kernel = AttentionCuda::build(info);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    // malloc
    auto &dev = *device::init(Device::Type::Nvidia, 0, "");
    auto qGpu = dev.malloc(q->bytesSize()),
         kGpu = dev.malloc(k->bytesSize()),
         vGpu = dev.malloc(v->bytesSize()),
         oGpu = dev.malloc(o->bytesSize()),
         workspace = dev.malloc(workspaceSize);
    // inference
    {
        void const *inputs[]{*qGpu, *kGpu, *vGpu};
        void *outputs[]{*oGpu};
        routine(res, *workspace, inputs, outputs);
    }
}

#endif
