#include "../../utilities/cuda/cublaslt_utils.cuh"
#include "cuda_kernel.hh"
#include "hardware/functions.h"
#include "kernel/cuda/reduce.cuh"

namespace refactor::kernel {
    using K = AttentionCuda;
    using namespace cublas;

    static __forceinline__ __device__ bool mask(int tokid, int posid) {
        return true;
    }

    // gridDim.x = batch * nHead
    // gridDim.y = seqLen
    template<class T, class Mask>
    static __global__ void softmax(
        T *__restrict__ attention,
        Mask mask,
        uint32_t seqLen,
        uint32_t bufLen) {
        // int offset = (blockIdx.x * len_q + blockIdx.y) * len_buf;
        // SharedMemory<float> shared;
        // float *smem = shared.getPointer();

        // for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        //     T pb = (position_bias == nullptr) ? T(0.) : position_bias[offset + i];
        //     smem[i] = mask[blockIdx.y * len_buf + i] > 0 ? x[offset + i] * scale + pb : -Inf<T>();
        // }
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
                        auto att = workspace;
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
                            att, mask, info.seqLen, info.seqLen);

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
