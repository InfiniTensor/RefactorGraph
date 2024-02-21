#include "../../utilities/cuda/cublaslt_utils.cuh"
#include "cuda_kernel.hh"
#include "hardware/functions.h"
#include "kernel/cuda/functions.cuh"
#include <cub/block/block_reduce.cuh>

namespace refactor::kernel {
    using K = AttentionCuda;
    using namespace cublas;

    // 因果系统的注意力遮罩。
    // tokenId: 第几个词
    //  seqLen: 此次处理的词数
    //   posId: 在 kv cache 中的位置
    //  attLen = pastSeqLen + seqLen
    struct AttentionCausualMask {
        __forceinline__ __device__ bool
        operator()(int tokenId, int seqLen,
                   int posId, int attLen) {
            // tokenId ↓ |<---attLen---->|
            //         0 | * * ... *     |
            //         1 | * * ... * *   |
            //         2 | * * ... * * * |
            // seqLen: 3 |---------------|
            return attLen + tokenId >= posId + seqLen;
        }
    };

    // gridDim.x = batch * nHead
    // gridDim.y = seqLen
    // blockDim.x = 1024
    // sizeof(shared) = attLen * sizeof(float)
    template<class T, class Mask>
    static __global__ void softmax(
        T *__restrict__ att,
        Mask mask,
        uint32_t attLen,
        uint32_t bufLen) {
        // 找到这个线程块对应的 attention 区域
        att += (blockIdx.x * gridDim.y + blockIdx.y) * bufLen;
        // 将输入装入共享内存并 cast + mask
        extern __shared__ float shared[];// size = attLen = pastSeqLen + seqLen
        for (auto i = threadIdx.x; i < attLen; i += blockDim.x) {
            shared[i] = mask(blockIdx.y, gridDim.y, i, attLen) ? float(att[i]) : -__FLT_MAX__;
        }

        using BlockReduce = cub::BlockReduce<float, 1024>;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        __shared__ float sharedMax, sharedSum;

        float localMax = -1e20;
        for (auto i = threadIdx.x; i < attLen; i += blockDim.x) {
            localMax = cub::Max()(localMax, shared[i]);
        }
        localMax = BlockReduce(tempStorage).Reduce(localMax, cub::Max(), attLen);
        if (threadIdx.x == 0) { sharedMax = localMax; }
        __syncthreads();

        float localSum = 1e-20;
        for (auto i = threadIdx.x; i < attLen; i += blockDim.x) {
            localSum += shared[i] = expf(shared[i] - sharedMax);
        }
        localSum = BlockReduce(tempStorage).Reduce(localSum, cub::Sum(), attLen);
        if (threadIdx.x == 0) { sharedSum = localSum; }
        __syncthreads();

        auto reciprocal = fdividef(1, sharedSum);
        for (auto i = threadIdx.x; i < attLen; i += blockDim.x) {
            att[i] = shared[i] * reciprocal;
        }
    }

    static __global__ void concatCache(
        void *__restrict__ cache,
        void const *__restrict__ value,
        dim_t pageStrideI,
        dim_t pageStrideO,
        dim_t pastOffset,
        dim_t n_items) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n_items) {
            auto dst = tid / pageStrideI * pageStrideO + pastOffset + (tid % pageStrideI);
            reinterpret_cast<float4 *>(cache)[dst] = reinterpret_cast<float4 const *>(value)[tid];
        }
    }
    constexpr uint64_t DYNAMIC_WORKSPACE_SIZE = 40 << 20;// 试出来 40MiB 是够用的

    RoutineWorkspace K::lower(Resources &res) const {
        auto handle = res.fetchOrStore<CublasLtContext>()->handle;

        constexpr auto ROW_MAJOR = CUBLASLT_ORDER_ROW;
        constexpr auto COL_MAJOR = CUBLASLT_ORDER_COL;

        if (!info.cacheLen) {
            if (info.nHead == info.nKVHead) {
                // RAII for closure
                struct Descriptors {
                    MatMulDescriptor mul;
                    MatrixDescriptor q, k, v, att;
                    cublasLtMatmulAlgo_t algoQK, algoAV;
                    size_t workspaceSizeQK, workspaceSizeAV;

                    Descriptors(CublasLtContext const &context,
                                AttentionInfo info)
                        : mul(computeTypeConvert(info.dataType),
                              dataTypeConvert(info.dataType)),
                          q(MatrixLayout{
                              .dataType = dataTypeConvert(info.dataType),
                              .rows = static_cast<uint64_t>(info.seqLen),
                              .cols = static_cast<uint64_t>(info.headDim),
                              .majorStride = static_cast<int64_t>(info.headDim),
                              .order = ROW_MAJOR,
                              .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                              .batchStride = static_cast<int64_t>(info.seqLen * info.headDim),
                          }),
                          k(MatrixLayout{
                              .dataType = dataTypeConvert(info.dataType),
                              .rows = static_cast<uint64_t>(info.headDim),
                              .cols = static_cast<uint64_t>(info.seqLen),
                              .majorStride = static_cast<int64_t>(info.headDim),
                              .order = COL_MAJOR,
                              .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                              .batchStride = static_cast<int64_t>(info.seqLen * info.headDim),
                          }),
                          v(MatrixLayout{
                              .dataType = dataTypeConvert(info.dataType),
                              .rows = static_cast<uint64_t>(info.seqLen),
                              .cols = static_cast<uint64_t>(info.headDim),
                              .majorStride = static_cast<int64_t>(info.headDim),
                              .order = ROW_MAJOR,
                              .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                              .batchStride = static_cast<int64_t>(info.seqLen * info.headDim),
                          }),
                          att(MatrixLayout{
                              .dataType = dataTypeConvert(info.dataType),
                              .rows = static_cast<uint64_t>(info.seqLen),
                              .cols = static_cast<uint64_t>(info.seqLen),
                              .majorStride = static_cast<int64_t>(info.seqLen),
                              .order = ROW_MAJOR,
                              .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                              .batchStride = static_cast<int64_t>(info.seqLen * info.seqLen),
                          }) {
                        auto [algoQK_, workspaceSizeQK_] = tune(context.handle, mul, q, k, att, DYNAMIC_WORKSPACE_SIZE);
                        auto [algoAV_, workspaceSizeAV_] = tune(context.handle, mul, att, v, q, DYNAMIC_WORKSPACE_SIZE);
                        algoQK = algoQK_;
                        algoAV = algoAV_;
                        workspaceSizeQK = workspaceSizeQK_;
                        workspaceSizeAV = workspaceSizeAV_;
                    }
                };

                auto const &context = *res.fetchOrStore<CublasLtContext>();
                auto d = std::make_shared<Descriptors>(context, info);
                auto workspaceSize = info.attSize(0);
                workspaceSize = hardware::alignBytes(workspaceSize, 256);
                workspaceSize += d->workspaceSizeQK;
                workspaceSize += d->workspaceSizeAV;
                workspaceSize = hardware::alignBytes(workspaceSize, 256);

                auto routine = [d = std::move(d), info = this->info]//
                    (Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                        auto handle = res.fetchOrStore<CublasLtContext>()->handle;
                        auto q = inputs[0];
                        auto k = inputs[1];
                        auto v = inputs[2];
                        auto o = outputs[0];
                        auto att = reinterpret_cast<half *>(workspace);
                        auto workspaceQK = reinterpret_cast<uint8_t *>(workspace) + hardware::alignBytes(info.attSize(0), 256);
                        auto workspaceAV = workspaceQK + hardware::alignBytes(d->workspaceSizeQK, 256);
                        auto stream = cudaStreamLegacy;
                        {
                            half alpha = rsqrtf(info.headDim), beta = 0;
                            cublasLtMatmul(
                                handle, d->mul.get(),
                                &alpha,
                                q, d->q.get(),
                                k, d->k.get(),
                                &beta,
                                att, d->att.get(),
                                att, d->att.get(),
                                &d->algoQK,
                                workspaceQK, d->workspaceSizeQK,
                                stream);
                        }
                        auto attLen = info.attLen(0);
                        auto bufLen = attLen;
                        softmax<<<dim3(info.batch * info.nHead, info.seqLen),
                                  std::min(1024u, attLen),
                                  attLen * sizeof(float),
                                  stream>>>(
                            att, AttentionCausualMask(), attLen, bufLen);
                        {
                            half alpha = 1, beta = 0;
                            cublasLtMatmul(
                                handle, d->mul.get(),
                                &alpha,
                                att, d->att.get(),
                                v, d->v.get(),
                                &beta,
                                o, d->q.get(),
                                o, d->q.get(),
                                &d->algoAV,
                                workspaceAV, d->workspaceSizeAV,
                                stream);
                        }
                    };

                return {std::move(routine), workspaceSize};
            }
            TODO("");
        }
        if (info.concatCache && !info.resetCache) {
            if (info.nHead == info.nKVHead) {

                // RAII for closure
                struct Descriptors {
                    MatMulDescriptor mul;

                    Descriptors(AttentionInfo info)
                        : mul(computeTypeConvert(info.dataType),
                              dataTypeConvert(info.dataType)) {}
                };

                auto const &context = *res.fetchOrStore<CublasLtContext>();
                auto d = std::make_shared<Descriptors>(info);
                auto attentionSize = info.maxAttSize();
                auto workspaceSize = DYNAMIC_WORKSPACE_SIZE + attentionSize;

                for (auto attLen = 0; attLen < 2048; ++attLen) {
                    MatrixDescriptor
                        q_(MatrixLayout{
                            .dataType = dataTypeConvert(info.dataType),
                            .rows = static_cast<uint64_t>(info.seqLen),
                            .cols = static_cast<uint64_t>(info.headDim),
                            .majorStride = static_cast<int64_t>(info.headDim),
                            .order = ROW_MAJOR,
                            .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                            .batchStride = static_cast<int64_t>(info.seqLen * info.headDim),
                        }),
                        k_(MatrixLayout{
                            .dataType = dataTypeConvert(info.dataType),
                            .rows = static_cast<uint64_t>(info.headDim),
                            .cols = static_cast<uint64_t>(attLen),
                            .majorStride = static_cast<int64_t>(info.headDim),
                            .order = COL_MAJOR,
                            .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                            .batchStride = static_cast<int64_t>(info.cacheLen * info.headDim),
                        }),
                        v_(MatrixLayout{
                            .dataType = dataTypeConvert(info.dataType),
                            .rows = static_cast<uint64_t>(attLen),
                            .cols = static_cast<uint64_t>(info.headDim),
                            .majorStride = static_cast<int64_t>(info.headDim),
                            .order = ROW_MAJOR,
                            .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                            .batchStride = static_cast<int64_t>(info.cacheLen * info.headDim),
                        }),
                        att_(MatrixLayout{
                            .dataType = dataTypeConvert(info.dataType),
                            .rows = static_cast<uint64_t>(info.seqLen),
                            .cols = static_cast<uint64_t>(attLen),
                            .majorStride = static_cast<int64_t>(info.cacheLen),
                            .order = ROW_MAJOR,
                            .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                            .batchStride = static_cast<int64_t>(info.cacheLen * info.seqLen),
                        });
                    tune(handle, d->mul,
                         q_, k_, att_,
                         DYNAMIC_WORKSPACE_SIZE);
                    tune(handle, d->mul,
                         att_, v_, q_,
                         DYNAMIC_WORKSPACE_SIZE);
                }

                auto routine = [d = std::move(d), info = this->info]//
                    (Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                        auto handle = res.fetchOrStore<CublasLtContext>()->handle;
                        auto q = inputs[0];
                        auto k = inputs[1];
                        auto v = inputs[2];
                        int64_t past;
                        cudaMemcpy(&past, inputs[3], sizeof(int64_t), cudaMemcpyDeviceToHost);
                        auto attLen = info.attLen(past);
                        auto o = reinterpret_cast<half *>(outputs[0]);
                        auto kCache = reinterpret_cast<half *>(outputs[1]);
                        auto vCache = reinterpret_cast<half *>(outputs[2]);
                        auto att = reinterpret_cast<half *>(reinterpret_cast<uint8_t *>(workspace) + DYNAMIC_WORKSPACE_SIZE);
                        auto stream = cudaStreamLegacy;
                        {
                            auto itemsPerLine = info.headDim * sizeof(half) / sizeof(float4);
                            auto threads = info.batch * info.nHead * info.seqLen * itemsPerLine;
                            auto blocks = (threads + 1023) / 1024;
                            concatCache<<<blocks, 1024, 0, stream>>>(
                                kCache, k,
                                info.seqLen * itemsPerLine,
                                info.cacheLen * itemsPerLine,
                                past * itemsPerLine,
                                threads);
                            concatCache<<<blocks, 1024, 0, stream>>>(
                                vCache, v,
                                info.seqLen * itemsPerLine,
                                info.cacheLen * itemsPerLine,
                                past * itemsPerLine,
                                threads);
                        }
                        MatrixDescriptor
                            q_(MatrixLayout{
                                .dataType = dataTypeConvert(info.dataType),
                                .rows = static_cast<uint64_t>(info.seqLen),
                                .cols = static_cast<uint64_t>(info.headDim),
                                .majorStride = static_cast<int64_t>(info.headDim),
                                .order = ROW_MAJOR,
                                .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                                .batchStride = static_cast<int64_t>(info.seqLen * info.headDim),
                            }),
                            k_(MatrixLayout{
                                .dataType = dataTypeConvert(info.dataType),
                                .rows = static_cast<uint64_t>(info.headDim),
                                .cols = static_cast<uint64_t>(attLen),
                                .majorStride = static_cast<int64_t>(info.headDim),
                                .order = COL_MAJOR,
                                .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                                .batchStride = static_cast<int64_t>(info.cacheLen * info.headDim),
                            }),
                            v_(MatrixLayout{
                                .dataType = dataTypeConvert(info.dataType),
                                .rows = static_cast<uint64_t>(attLen),
                                .cols = static_cast<uint64_t>(info.headDim),
                                .majorStride = static_cast<int64_t>(info.headDim),
                                .order = ROW_MAJOR,
                                .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                                .batchStride = static_cast<int64_t>(info.cacheLen * info.headDim),
                            }),
                            att_(MatrixLayout{
                                .dataType = dataTypeConvert(info.dataType),
                                .rows = static_cast<uint64_t>(info.seqLen),
                                .cols = static_cast<uint64_t>(attLen),
                                .majorStride = static_cast<int64_t>(info.cacheLen),
                                .order = ROW_MAJOR,
                                .batchCount = static_cast<int32_t>(info.batch * info.nHead),
                                .batchStride = static_cast<int64_t>(info.cacheLen * info.seqLen),
                            });
                        {
                            auto [algo, workspaceSize] = tune(
                                handle, d->mul,
                                q_, k_, att_,
                                DYNAMIC_WORKSPACE_SIZE);
                            half alpha = rsqrtf(info.headDim), beta = 0;
                            cublasLtMatmul(
                                handle, d->mul.get(),
                                &alpha,
                                q, q_.get(),
                                kCache, k_.get(),
                                &beta,
                                att, att_.get(),
                                att, att_.get(),
                                &algo,
                                workspace, workspaceSize,
                                stream);
                        }
                        softmax<<<dim3(info.batch * info.nHead, info.seqLen),
                                  std::min(1024u, attLen),
                                  attLen * sizeof(float),
                                  stream>>>(
                            att, AttentionCausualMask(), attLen, info.cacheLen);
                        {
                            auto [algo, workspaceSize] = tune(
                                handle, d->mul,
                                att_, v_, q_,
                                DYNAMIC_WORKSPACE_SIZE);
                            half alpha = 1, beta = 0;
                            cublasLtMatmul(
                                handle, d->mul.get(),
                                &alpha,
                                att, att_.get(),
                                vCache, v_.get(),
                                &beta,
                                o, q_.get(),
                                o, q_.get(),
                                &algo,
                                workspace, workspaceSize,
                                stream);
                        }
                    };

                return {std::move(routine), workspaceSize};
            }
            TODO("");
        }

        TODO("");
    }

}// namespace refactor::kernel
