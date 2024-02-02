#include "../../utilities/cuda/cublaslt_utils.cuh"
#include "cuda_kernel.hh"
#include "hardware/functions.h"
#include "kernel/cuda/reduce.cuh"

namespace refactor::kernel {
    using K = AttentionCuda;
    using namespace cublas;

    // 因果系统的注意力遮罩。
    // tokenId: 第几个词
    //  seqLen: 此次处理的词数
    //   posId: 在 kv cache 中的位置
    //  attLen = pastSeqLen + seqLen
    static __forceinline__ __device__ bool
    causualMask(int tokenId, int seqLen,
                int posId, int attLen) {
        // tokenId ↓ |<---attLen---->|
        //         0 | * * ... *     |
        //         1 | * * ... * *   |
        //         2 | * * ... * * * |
        // seqLen: 3 |---------------|
        return attLen + tokenId >= posId + seqLen;
    }

    // gridDim.x = batch * nHead
    // gridDim.y = seqLen
    // blockDim.x = min(1024, attLen)
    template<class T>
    static __global__ void softmax(
        T *__restrict__ att,
        bool (*mask)(int, int, int, int),
        uint32_t attLen,
        uint32_t bufLen) {
        // 找到这个线程块对应的 attention 区域
        att += (blockIdx.x * gridDim.x + gridDim.y) * bufLen;
        // 将输入装入共享内存并 cast + mask
        extern __shared__ float shared[];// size = attLen = pastSeqLen + seqLen
        for (auto i = threadIdx.x; i < attLen; i += blockDim.x) {
            shared[i] = mask(blockIdx.y, gridDim.y, i, attLen) ? float(att[i]) : -__FLT_MAX__;
        }

        // float local_max = -1e20;
        // for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        //     local_max = fmaxf(local_max, smem[i]);
        // }
        // local_max = functions::blockReduceMax<float>(local_max);

        // float local_sum = 1e-20;
        // for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        //     float v = expf(float(smem[i]) - local_max);
        //     smem[i] = v;
        //     local_sum += v;
        // }
        // local_sum = functions::blockReduceSum<float>(local_sum);
        // for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        //     x[offset + i] = float(smem[i]) / local_sum;
        // }
    }

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
                    size_t attSize, workspaceSizeQK, workspaceSizeAV;

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
                          }),
                          attSize(info.batch * info.nHead * info.seqLen * info.seqLen * info.dataType.size()) {
                        auto [algoQK_, workspaceSizeQK_] = tune(context.handle, mul, q, k, att);
                        auto [algoAV_, workspaceSizeAV_] = tune(context.handle, mul, att, v, q);
                        algoQK = algoQK_;
                        algoAV = algoAV_;
                        workspaceSizeQK = workspaceSizeQK_;
                        workspaceSizeAV = workspaceSizeAV_;
                    }
                };

                auto const &context = *res.fetchOrStore<CublasLtContext>();
                auto d = std::make_shared<Descriptors>(context, info);
                auto workspaceSize = d->attSize;
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
                        auto workspaceQK = reinterpret_cast<uint8_t *>(workspace) + hardware::alignBytes(d->attSize, 256);
                        auto workspaceAV = workspaceQK + hardware::alignBytes(d->workspaceSizeQK, 256);

                        float alpha = 1, beta = 0;
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
                            cudaStreamLegacy);

                        softmax<<<dim3(info.batch * info.nHead, info.seqLen), info.seqLen>>>(
                            att, causualMask, info.seqLen, info.seqLen);

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
                            cudaStreamLegacy);
                    };
                return {std::move(routine), workspaceSize};
            }
        }
        TODO("");
    }

}// namespace refactor::kernel
