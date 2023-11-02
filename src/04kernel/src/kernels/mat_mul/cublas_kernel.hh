#ifndef KERNEL_MATMUL_CUBLAS_KERNEL_HH
#define KERNEL_MATMUL_CUBLAS_KERNEL_HH

#include "kernel/attributes/matmul_info.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

#ifdef USE_CUDA
#include <cublas_v2.h>

constexpr int N_ALGO = 24;
constexpr int ALGOS[N_ALGO] = {
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_ALGO18,
    CUBLAS_GEMM_ALGO19,
    CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21,
    CUBLAS_GEMM_ALGO22,
    CUBLAS_GEMM_ALGO23,
};
#endif

namespace refactor::kernel {

    struct MatMulCublas final : public Kernel {
#ifdef USE_CUDA
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#endif

        MatMulInfo info;

        explicit MatMulCublas(MatMulInfo) noexcept;

        static KernelBox build(Tensor const &, Tensor const &, Tensor const &, MatMulInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_CUBLAS_KERNEL_HH
