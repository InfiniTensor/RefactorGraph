#include "../../utilities/cuda/cublas_context.hh"
#include "../expand/cuda_kernel.hh"
#include "cublas_kernel.hh"
#include <cublas_v2.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T>
    Routine lowerTyped(cudaDataType_t cudaDataType, MatMulInfo info, Resources &res) noexcept {
        return [cudaDataType,
                alpha = static_cast<T>(info.alpha),
                beta = static_cast<T>(info.biasExpand ? info.beta : 0.0f),
                tA = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                tB = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                m = info.m, n = info.n, k = info.k,
                strideY = info.m * info.n,
                strideA = info.m * info.k,
                strideB = info.k * info.n,
                lda = info.transA ? info.m : info.k,
                ldb = info.transB ? info.k : info.n,
                biasEx = info.biasExpand
                             ? std::make_optional(ExpandCuda(*info.biasExpand).lower(res))
                             : std::nullopt,
                broadcaster = info.broadcaster](Resources &res, void const **inputs, void **outputs) {
            if (biasEx) { (*biasEx)(res, inputs + 2, outputs); }

            auto handle = res.fetchOrStore<CublasContext>()->handle;
            auto a = reinterpret_cast<T const *>(inputs[0]);
            auto b = reinterpret_cast<T const *>(inputs[1]);
            auto y = reinterpret_cast<T *>(outputs[0]);
            uint32_t offset[2];
            for (auto i : range0_(broadcaster.outputsCount)) {
                broadcaster.locate(i, offset);
                auto stat = cublasGemmEx(
                    handle, tB, tA, n, m, k, &alpha, b + strideB * offset[1],
                    cudaDataType, ldb, a + strideA * offset[0], cudaDataType, lda, &beta, y + strideY * i,
                    cudaDataType, n, cudaDataType, CUBLAS_GEMM_DEFAULT);
            }
        };
    }

    Routine MatMulCublas::lower(Resources &res) const noexcept {
        res.fetchOrStore<CublasContext>();
        switch (info.dataType) {
            case DataType::F32:
                return lowerTyped<float>(CUDA_R_32F, info, res);
            case DataType::F64:
                return lowerTyped<double>(CUDA_R_64F, info, res);
            case DataType::FP16:
                return lowerTyped<half>(CUDA_R_16F, info, res);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
