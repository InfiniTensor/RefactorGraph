#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T> __device__ __forceinline__ static int8_t sub(T, T);
    template<> __device__ __forceinline__ int8_t sub<int8_t>(int8_t a, int8_t b) { return a - b; }
    template<> __device__ __forceinline__ int8_t sub<uint8_t>(uint8_t a, uint8_t b) { return static_cast<int8_t>(static_cast<int16_t>(a) - static_cast<int16_t>(b)); }

    template<class T>
    struct MatMulIntegerZPFunctor {
        dim_t groupSize;
        T const *src, *zp;

        __device__ int8_t operator()(size_t i) const noexcept {
            return sub(src[i], zp[i / groupSize]);
        }
    };

    template<class T>
    static void applyZeroPoint(MatMulIntegerInfo::Input meta, int8_t *dst, void const *src, void const *zp) {
        thrust::tabulate(
            thrust::device,
            dst, dst + meta.groupCount * meta.groupSize,
            MatMulIntegerZPFunctor<T>{
                .groupSize = meta.groupSize,
                .src = reinterpret_cast<T const *>(src),
                .zp = reinterpret_cast<T const *>(zp),
            });
    }

    auto MatMulIntegerCublas::lower(Resources &res) const noexcept -> RoutineWorkspace {

        size_t workspace = 0;
        if (info.a.withZeroPoint) {
            workspace += info.a.groupCount * info.a.groupSize;
        }
        if (info.b.withZeroPoint) {
            workspace += info.b.groupCount * info.b.groupSize;
        }

        auto routine = [info = info](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto workspacePtr = reinterpret_cast<int8_t *>(workspace);
            auto a = reinterpret_cast<int8_t const *>(inputs[0]),
                 b = reinterpret_cast<int8_t const *>(inputs[1]);
            auto y = reinterpret_cast<int32_t *>(outputs[0]);

            if (auto meta = info.a; meta.withZeroPoint) {
                if (meta.signed_) {
                    applyZeroPoint<int8_t>(meta, workspacePtr, a, inputs[2]);
                } else {
                    applyZeroPoint<uint8_t>(meta, workspacePtr, a, inputs[2]);
                }
                a = workspacePtr;
                workspacePtr += meta.groupCount * meta.groupSize;
            }
            if (auto meta = info.b; meta.withZeroPoint) {
                if (meta.signed_) {
                    applyZeroPoint<int8_t>(meta, workspacePtr, b, inputs[3]);
                } else {
                    applyZeroPoint<uint8_t>(meta, workspacePtr, b, inputs[3]);
                }
                b = workspacePtr;
            }

            int32_t alpha = 1, beta = 0;
            auto m = info.m,
                 n = info.n,
                 k = info.k;
            auto strideY = m * n,
                 strideA = m * k,
                 strideB = k * n;
            auto lda = info.k,
                 ldb = info.n;
            if (info.broadcaster.needBroadcast()) {

                uint32_t offset[2];
                for (auto i : range0_(info.broadcaster.outputsCount)) {
                    info.broadcaster.locate(i, offset);
                    cublasGemmEx(
                        res.fetchOrStore<CublasContext>()->handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha,
                        b + strideB * offset[1], CUDA_R_8I, ldb,
                        a + strideA * offset[0], CUDA_R_8I, lda,
                        &beta, y + strideY * i, CUDA_R_32I,
                        n, CUDA_R_32I,
                        CUBLAS_GEMM_DEFAULT);
                }
            } else {

                cublasGemmStridedBatchedEx(
                    res.fetchOrStore<CublasContext>()->handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, CUDA_R_8I, ldb, strideB,
                    a, CUDA_R_8I, lda, strideA,
                    &beta, y, CUDA_R_32I,
                    n, m * n, info.broadcaster.outputsCount, CUDA_R_32I,
                    CUBLAS_GEMM_DEFAULT);
            }
        };

        res.fetchOrStore<CublasContext>();
        return {std::move(routine), workspace};
    }

}// namespace refactor::kernel
