#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T> __device__ __forceinline__ static float sub(T a, T b) {
        return static_cast<float>(a) - static_cast<float>(b);
    }

    template<class T>
    struct MatMulIntegerZPFunctorScalar {
        T const *zp;

        __device__ float operator()(T x) const noexcept {
            return sub(x, *zp);
        }
    };

    template<class T>
    static void applyZeroPointScalar(
        size_t size, void *dst_, void const *src_, void const *zp_) {

        auto dst = reinterpret_cast<float *>(dst_);
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

        __device__ float operator()(size_t idx) const noexcept {
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
        void *dst_, void const *src_, void const *zp_) {

        auto dst = reinterpret_cast<float *>(dst_);
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
        void *dst_, void const *src_, void const *zp_) {

        auto dst = reinterpret_cast<float *>(dst_);
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

    template<class To, class From>
    static void applyCast(
        void *dst_, void const *src_, size_t size) {

        auto dst = reinterpret_cast<To *>(dst_);
        auto src = reinterpret_cast<From const *>(src_);
        thrust::transform(thrust::device,
                          src, src + size,
                          dst, cub::CastOp<To>{});
    }

    auto MatMulIntegerCublas::lower(Resources &res) const noexcept -> RoutineWorkspace {

        size_t workspace = 0;
        workspace += info.batch() * info.m * info.k * sizeof(float);
        workspace += info.batch() * info.k * info.n * sizeof(float);
        workspace += info.batch() * info.m * info.n * sizeof(float);

        auto routine = [info = info](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto workspacePtr = reinterpret_cast<int8_t *>(workspace);
            float *a, *b, *y;

            {
                auto size = info.batch() * info.m * info.k;
                auto input = inputs[0];
                auto zp = inputs[2];
                a = reinterpret_cast<float *>(workspacePtr);
                workspacePtr += size * sizeof(float);

                if (auto meta = info.a; meta.withZeroPoint) {
                    if (meta.scalar) {
                        if (meta.signed_) {
                            applyZeroPointScalar<int8_t>(size, a, input, zp);
                        } else {
                            applyZeroPointScalar<uint8_t>(size, a, input, zp);
                        }
                    } else {
                        if (meta.signed_) {
                            applyZeroPointA<int8_t>(info.batch(), info.m, info.k, a, input, zp);
                        } else {
                            applyZeroPointA<uint8_t>(info.batch(), info.m, info.k, a, input, zp);
                        }
                    }
                } else {
                    if (meta.signed_) {
                        applyCast<float, int8_t>(a, input, size);
                    } else {
                        applyCast<float, uint8_t>(a, input, size);
                    }
                }
            }
            {
                auto size = info.batch() * info.k * info.n;
                auto input = inputs[1];
                auto zp = inputs[3];
                b = reinterpret_cast<float *>(workspacePtr);
                workspacePtr += size * sizeof(float);

                if (auto meta = info.b; meta.withZeroPoint) {
                    if (meta.scalar) {
                        if (meta.signed_) {
                            applyZeroPointScalar<int8_t>(size, b, input, zp);
                        } else {
                            applyZeroPointScalar<uint8_t>(size, b, input, zp);
                        }
                    } else {
                        if (meta.signed_) {
                            applyZeroPointA<int8_t>(info.batch(), info.m, info.k, b, input, zp);
                        } else {
                            applyZeroPointA<uint8_t>(info.batch(), info.m, info.k, b, input, zp);
                        }
                    }
                } else {
                    if (meta.signed_) {
                        applyCast<float, int8_t>(b, input, size);
                    } else {
                        applyCast<float, uint8_t>(b, input, size);
                    }
                }
            }
            y = reinterpret_cast<float *>(workspacePtr);

            auto handle = res.fetchOrStore<CublasContext>()->handle;
            float alpha = 1,
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
                        b + strideB * offset[1], CUDA_R_32F, ldb,
                        a + strideA * offset[0], CUDA_R_32F, lda,
                        &beta, y + strideY * i, CUDA_R_32F, n,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                }
            } else {

                CUBLAS_ASSERT(cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, CUDA_R_32F, ldb, strideB,
                    a, CUDA_R_32F, lda, strideA,
                    &beta, y, CUDA_R_32F, n,
                    strideY, info.batch(),
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
            }

            applyCast<int32_t, float>(outputs[0], y, info.batch() * info.m * info.n);
        };

        res.fetchOrStore<CublasContext>();
        return {std::move(routine), workspace};
    }

}// namespace refactor::kernel
