#ifdef USE_CUDA

#include "../../../src/kernels/attention/cuda_kernel.hh"
#include "hardware/device_manager.h"
#include "kernel/cuda/functions.cuh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, AttentionCudaNoKvCache) {
    // build routine
    AttentionInfo info{
        .dataType = DataType::F32,
        .batch = 1,
        .nHead = 4,
        .nKVHead = 4,
        .seqLen = 31,
        .headDim = 256,
        .cacheLen = 0,
        .concatCache = false,
        .resetCache = false,
    };
    auto q = Tensor::share(DataType::F32, Shape{info.batch, info.nHead, info.seqLen, info.headDim}),
         k = Tensor::share(DataType::F32, Shape{info.batch, info.nKVHead, info.seqLen, info.headDim}),
         v = Tensor::share(DataType::F32, Shape{info.batch, info.nKVHead, info.seqLen, info.headDim}),
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
    // put input data
    std::vector<float>
        q_(q->elementsSize(), 1),
        k_(k->elementsSize(), 1),
        v_(v->elementsSize(), 1),
        o_(o->elementsSize());
    qGpu->copyFromHost(q_.data());
    kGpu->copyFromHost(k_.data());
    vGpu->copyFromHost(v_.data());
    // inference
    {
        void const *inputs[]{*qGpu, *kGpu, *vGpu};
        void *outputs[]{*oGpu};
        routine(res, *workspace, inputs, outputs);
    }
    cuda::sync();
}

#endif
