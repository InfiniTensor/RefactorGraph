#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;
    using DT = DataType;

    template<typename T>
    struct MatMulBroadcastBiasFunctor {
        T const *C;
        T *Y;
        size_t const n;
        size_t const strideC0, strideC1;

        __device__ void operator()(size_t t) const noexcept {
            size_t i = t / n;
            size_t j = t % n;
            memcpy(Y + t, C + i * strideC0 + j * strideC1, sizeof(T));
        }
    };

    template<typename T>
    struct MatMulCopyBiasFunctor {
        T *dst;
        T const *src;
        size_t stride;

        __device__ void operator()(size_t i) const noexcept {
            memcpy(dst + i * stride, src, stride * sizeof(T));
        }
    };

    template<class T, cudaDataType_t cudaDataType>
    Routine lowerTyped(MatMulInfo info, size_t strideC0, size_t strideC1) noexcept {
        return [alpha = static_cast<T>(info.alpha),
                beta = static_cast<T>(info.biasType != BiasType::NoBias ? info.beta : 0.0f),
                tA = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                tB = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                m = info.m, n = info.n, k = info.k, batch = info.batch(),
                strideY = info.m * info.n,
                strideA = info.m * info.k,
                strideB = info.k * info.n,
                strideC0, strideC1,
                lda = info.transA ? info.m : info.k,
                ldb = info.transB ? info.k : info.n,
                broadcaster = info.broadcaster](Resources &res, void const **inputs, void **outputs) {
            auto A = reinterpret_cast<T const *>(inputs[0]);
            auto B = reinterpret_cast<T const *>(inputs[1]);
            auto Y = reinterpret_cast<T *>(outputs[0]);

            if (beta != T{}) {
                auto C = reinterpret_cast<T const *>(inputs[2]);
                // Expand bias to 2D and store in final output Y
                thrust::for_each_n(thrust::device,
                                   thrust::counting_iterator<size_t>(0), strideY,
                                   MatMulBroadcastBiasFunctor<T>{C, Y, n, strideC0, strideC1});
                // Copy 2D bias to each batch
                if (batch > 1) {
                    thrust::for_each_n(thrust::device,
                                       thrust::counting_iterator<size_t>(1), batch,
                                       MatMulCopyBiasFunctor<T>{Y, Y, strideY});
                }
            }

            auto handle = res.fetchOrStore<CublasContext>()->handle;
            uint32_t offset[2];
            for (size_t i = 0; i < batch; i++) {
                broadcaster.locate(i, offset);
                auto stat = cublasGemmEx(
                    handle, tB, tA, n, m, k, &alpha, B + strideB * offset[1],
                    cudaDataType, ldb, A + strideA * offset[0], cudaDataType, lda, &beta, Y + strideY * i,
                    cudaDataType, n, cudaDataType, CUBLAS_GEMM_DEFAULT);
            }
        };
    }

    Routine MatMulCublas::lower(Resources &res) const noexcept {
        size_t strideC0 = 0, strideC1 = 0;
        switch (info.biasType) {
            case BiasType::NoBias:
            case BiasType::Scalar:
                break;
            case BiasType::RowVector:
                strideC1 = 1;
                break;
            case BiasType::ColVector:
                strideC0 = 1;
                break;
            case BiasType::Matrix:
                strideC1 = 1;
                strideC0 = info.n;
                break;
            default:
                UNREACHABLE();
        }

        res.fetchOrStore<CublasContext>();
        switch (info.dataType) {
            case DT::F32:
                return lowerTyped<float, CUDA_R_32F>(info, strideC0, strideC1);
            case DT::F64:
                return lowerTyped<double, CUDA_R_64F>(info, strideC0, strideC1);
            case DT::FP16:
                return lowerTyped<fp16_t, CUDA_R_16F>(info, strideC0, strideC1);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
