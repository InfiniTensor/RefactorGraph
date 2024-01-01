#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T> __device__ __forceinline__ static int8_t sub(T, T);
    template<> __device__ __forceinline__ int8_t sub<int8_t>(int8_t a, int8_t b) { return a - b; }
    template<> __device__ __forceinline__ int8_t sub<uint8_t>(uint8_t a, uint8_t b) {
        constexpr static int16_t MAX = 127;
        return static_cast<int8_t>(CUB_MIN(MAX, static_cast<int16_t>(a) - static_cast<int16_t>(b)));
    }

    template<class T>
    struct MatMulIntegerZPFunctorScalar {
        T const *zp;

        __device__ int8_t operator()(T x) const noexcept {
            return sub(x, *zp);
        }
    };

    template<class T>
    static void applyZeroPointScalar(
        size_t size, int8_t *dst, void const *src_, void const *zp_) {

        auto src = reinterpret_cast<T const *>(src_),
             zp = reinterpret_cast<T const *>(zp_);
        thrust::transform(thrust::device,
                          src, src + size,
                          dst, MatMulIntegerZPFunctorScalar<T>{zp});
    }

    template<class T>
    struct MatMulIntegerZPFunctor {
        dim_t m, n, a, b, c;
        T const *src, *zp;

        __device__ int8_t operator()(size_t idx) const noexcept {
            auto
                k = idx % n,
                j = idx / n % m,
                i = idx / n / m;
            return sub(src[idx], zp[i * a + j * b + k * c]);
        }
    };

    template<class T>
    static void applyZeroPointA(
        dim_t b, dim_t m, dim_t n,
        int8_t *dst, void const *src_, void const *zp_) {
        thrust::tabulate(thrust::device,
                         dst, dst + b * m * n,
                         MatMulIntegerZPFunctor<T>{
                             m,
                             n,
                             m,
                             1,
                             0,
                             reinterpret_cast<T const *>(src_),
                             reinterpret_cast<T const *>(zp_),
                         });
    }

    template<class T>
    static void applyZeroPointB(
        dim_t b, dim_t m, dim_t n,
        int8_t *dst, void const *src_, void const *zp_) {

        thrust::tabulate(thrust::device,
                         dst, dst + b * m * n,
                         MatMulIntegerZPFunctor<T>{
                             m,
                             n,
                             n,
                             0,
                             1,
                             reinterpret_cast<T const *>(src_),
                             reinterpret_cast<T const *>(zp_),
                         });
    }

    struct MatMulIntegerCastFunctor {
        __device__ int8_t operator()(uint8_t x) const noexcept {
            return static_cast<int8_t>(CUB_MIN(127, x));
        }
    };

    static void applyCast(
        size_t size, int8_t *dst, void const *src_) {

        auto src = reinterpret_cast<uint8_t const *>(src_);
        thrust::transform(thrust::device,
                          src, src + size,
                          dst, MatMulIntegerCastFunctor{});
    }

    auto MatMulIntegerCublas::lower(Resources &res) const noexcept -> RoutineWorkspace {

        size_t workspace = 0;
        if (info.a.withZeroPoint || !info.a.signed_) {
            workspace += info.batch() * info.m * info.k;
        }
        if (info.b.withZeroPoint || !info.b.signed_) {
            workspace += info.batch() * info.k * info.n;
        }

        auto routine = [info = info](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto workspacePtr = reinterpret_cast<int8_t *>(workspace);
            auto a = reinterpret_cast<int8_t const *>(inputs[0]),
                 b = reinterpret_cast<int8_t const *>(inputs[1]);
            auto y = reinterpret_cast<int32_t *>(outputs[0]);

            if (auto meta = info.a; meta.withZeroPoint) {
                auto size = info.batch() * info.m * info.k;
                auto zp = inputs[2];
                if (meta.scalar) {
                    if (meta.signed_) {
                        applyZeroPointScalar<int8_t>(size, workspacePtr, a, zp);
                    } else {
                        applyZeroPointScalar<uint8_t>(size, workspacePtr, a, zp);
                    }
                } else {
                    if (meta.signed_) {
                        applyZeroPointA<int8_t>(info.batch(), info.m, info.k, workspacePtr, a, zp);
                    } else {
                        applyZeroPointA<uint8_t>(info.batch(), info.m, info.k, workspacePtr, a, zp);
                    }
                }
                a = workspacePtr;
                workspacePtr += size;
            } else if (!meta.signed_) {
                auto size = info.batch() * info.m * info.k;
                applyCast(size, workspacePtr, a);
                a = workspacePtr;
                workspacePtr += size;
            }
            if (auto meta = info.b; meta.withZeroPoint) {
                auto size = info.batch() * info.k * info.n;
                auto zp = inputs[3];
                if (meta.scalar) {
                    if (meta.signed_) {
                        applyZeroPointScalar<int8_t>(size, workspacePtr, b, zp);
                    } else {
                        applyZeroPointScalar<uint8_t>(size, workspacePtr, b, zp);
                    }
                } else {
                    if (meta.signed_) {
                        applyZeroPointA<int8_t>(info.batch(), info.k, info.n, workspacePtr, b, zp);
                    } else {
                        applyZeroPointA<uint8_t>(info.batch(), info.k, info.n, workspacePtr, b, zp);
                    }
                }
                b = workspacePtr;
            } else if (!meta.signed_) {
                auto size = info.batch() * info.k * info.n;
                applyCast(size, workspacePtr, b);
                b = workspacePtr;
            }

            auto handle = res.fetchOrStore<CublasContext>()->handle;
            int32_t alpha = 1,
                    beta = 0;
            auto m = info.m,
                 n = info.n,
                 k = info.k;
            auto strideY = m * n,
                 strideA = m * k,
                 strideB = k * n;
            auto lda = k,
                 ldb = n;
            if (info.broadcaster.needBroadcast()) {

                uint32_t offset[2];
                for (auto i : range0_(info.batch())) {
                    info.broadcaster.locate(i, offset);
                    CUBLAS_ASSERT(cublasGemmEx(
                        handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha,
                        b + strideB * offset[1], CUDA_R_8I, ldb,
                        a + strideA * offset[0], CUDA_R_8I, lda,
                        &beta, y + strideY * i, CUDA_R_32I, n,
                        CUDA_R_32I, CUBLAS_GEMM_DEFAULT));
                }
            } else {

                CUBLAS_ASSERT(cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, CUDA_R_8I, ldb, strideB,
                    a, CUDA_R_8I, lda, strideA,
                    &beta, y, CUDA_R_32I, n,
                    strideY, info.batch(),
                    CUDA_R_32I, CUBLAS_GEMM_DEFAULT));
            }
        };

        res.fetchOrStore<CublasContext>();
        return {std::move(routine), workspace};
    }

}// namespace refactor::kernel
