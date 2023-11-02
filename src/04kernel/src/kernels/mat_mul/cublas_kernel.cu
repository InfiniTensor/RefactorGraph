#include "../../utilities/cuda/cublas_context.hh"
#include "cublas_kernel.hh"
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>


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

template<typename T>
void computeWithBias(cublasHandle_t handle, T const *A, T const *B, T const *C, T *Y,
                     T const alpha, T const beta,
                     cublasOperation_t tA, cublasOperation_t tB,
                     size_t m, size_t n, size_t k, size_t batch,
                     size_t strideY, size_t strideA, size_t strideB, size_t strideC0, size_t strideC1,
                     size_t lda, size_t ldb,
                     cublasGemmAlgo_t algo, cudaDataType_t cudaDataType,
                     refactor::kernel::Broadcaster broadcaster) {

    // Expand bias to 2D and store in final output Y
    thrust::for_each_n(thrust::device,
                       thrust::counting_iterator<size_t>(0), strideY,
                       MatMulBroadcastBiasFunctor<T>{C, Y, n, strideC0, strideC1});
    // Copy 2D bias to each batch
    if (batch > 1)
        thrust::for_each_n(thrust::device,
                           thrust::counting_iterator<size_t>(1), batch,
                           MatMulCopyBiasFunctor<T>{Y, Y, strideY});

    cublasStatus_t stat;
    uint32_t offset[2];
    for (size_t i = 0; i < batch; i++) {
        broadcaster.locate(i, offset);
        stat = cublasGemmEx(
            handle, tA, tB, n, m, k, &alpha, B + strideB * offset[1],
            cudaDataType, ldb, A + strideA * offset[0], cudaDataType, lda, &beta, Y + strideY * i,
            cudaDataType, n, cudaDataType, algo);
    }
}

template<typename T>
void computeNoBias(cublasHandle_t handle, T const *A, T const *B, T *Y,
                   T const alpha,
                   cublasOperation_t tA, cublasOperation_t tB,
                   size_t m, size_t n, size_t k, size_t batch,
                   size_t strideY, size_t strideA, size_t strideB,
                   size_t lda, size_t ldb,
                   cublasGemmAlgo_t algo, cudaDataType_t cudaDataType,
                   refactor::kernel::Broadcaster broadcaster) {
    T const beta{};
    cublasStatus_t stat;
    uint32_t offset[2];
    for (size_t i = 0; i < batch; i++) {
        broadcaster.locate(i, offset);
        stat = cublasGemmEx(
            handle, tA, tB, n, m, k, &alpha, B + strideB * offset[1],
            cudaDataType, ldb, A + strideA * offset[0], cudaDataType, lda, &beta, Y + strideY * i,
            cudaDataType, n, cudaDataType, algo);
    }
}


namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;
    using DT = DataType;

#define CASE(T, T_CUDA)                                                                                   \
    case DT::T: {                                                                                         \
        using T_ = primitive_t<DT::T>::type;                                                              \
        cudaDataType_t cudaDataType = T_CUDA;                                                             \
        if (info.biasType != BiasType::NoBias) {                                                          \
            return [alpha = static_cast<T_>(info.alpha), beta = static_cast<T_>(info.beta),               \
                    tA = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,                                         \
                    tB = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N,                                         \
                    m = info.m, n = info.n, k = info.k, batch = info.batch(),                             \
                    strideY = info.m * info.n,                                                            \
                    strideA = info.m * info.k,                                                            \
                    strideB = info.k * info.n,                                                            \
                    strideC0, strideC1,                                                                   \
                    lda = info.transA ? info.m : info.k,                                                  \
                    ldb = info.transB ? info.k : info.n,                                                  \
                    algo_ = algo,                                                                         \
                    cudaDataType,                                                                         \
                    broadcaster = info.broadcaster,                                                       \
                    compute = computeWithBias<T_>](Resources &res, void const **inputs, void **outputs) { \
                auto A = reinterpret_cast<T_ const *>(inputs[0]);                                         \
                auto B = reinterpret_cast<T_ const *>(inputs[1]);                                         \
                auto C = reinterpret_cast<T_ const *>(inputs[2]);                                         \
                auto Y = reinterpret_cast<T_ *>(outputs[0]);                                              \
                auto handle = res.fetchOrStore<CublasContext>() -> handle;                                \
                compute(handle, A, B, C, Y, alpha, beta, tA, tB,                                          \
                        m, n, k, batch, strideY, strideA, strideB, strideC0, strideC1, lda, ldb,          \
                        algo_, cudaDataType, broadcaster);                                                \
            };                                                                                            \
        } else {                                                                                          \
            return [alpha = static_cast<T_>(info.alpha),                                                  \
                    tA = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,                                         \
                    tB = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N,                                         \
                    m = info.m, n = info.n, k = info.k, batch = info.batch(),                             \
                    strideY = info.m * info.n,                                                            \
                    strideA = info.m * info.k,                                                            \
                    strideB = info.k * info.n,                                                            \
                    lda = info.transA ? info.m : info.k,                                                  \
                    ldb = info.transB ? info.k : info.n,                                                  \
                    algo_ = algo,                                                                         \
                    cudaDataType,                                                                         \
                    broadcaster = info.broadcaster,                                                       \
                    compute = computeNoBias<T_>](Resources &res, void const **inputs, void **outputs) {   \
                auto A = reinterpret_cast<T_ const *>(inputs[0]);                                         \
                auto B = reinterpret_cast<T_ const *>(inputs[1]);                                         \
                auto Y = reinterpret_cast<T_ *>(outputs[0]);                                              \
                auto handle = res.fetchOrStore<CublasContext>() -> handle;                                \
                compute(handle, A, B, Y, alpha, tA, tB,                                                   \
                        m, n, k, batch, strideY, strideA, strideB, lda, ldb,                              \
                        algo_, cudaDataType, broadcaster);                                                \
            };                                                                                            \
        }                                                                                                 \
    }


    auto
    MatMulCublas::lower() const noexcept -> Routine {
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

        switch (info.dataType) {
            CASE(F32, CUDA_R_32F);
            CASE(F64, CUDA_R_64F);
            CASE(FP16, CUDA_R_16F);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
