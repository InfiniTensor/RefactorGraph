#include "../../utilities/cuda/cublas_context.hh"
#include "../expand/cuda_kernel.hh"
#include "cublas_kernel.hh"
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T>
    Routine lowerTyped(cudaDataType_t cudaDataType, MatMulInfo info, Resources &res, std::optional<ExpandInfo> biasExpand, size_t strideC0, size_t strideC1) noexcept {
        auto biasEx = biasExpand
                          ? std::make_optional(ExpandCuda(*biasExpand).lower(res))
                          : std::nullopt;
        return [cudaDataType,
                alpha = static_cast<T>(info.alpha),
                beta = static_cast<T>(info.biasType != BiasType::NoBias ? info.beta : 0.0f),
                tA = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                tB = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                m = info.m, n = info.n, k = info.k,
                strideY = info.m * info.n,
                strideA = info.m * info.k,
                strideB = info.k * info.n,
                strideC0, strideC1,
                lda = info.transA ? info.m : info.k,
                ldb = info.transB ? info.k : info.n,
                biasEx,
                broadcaster = info.broadcaster](Resources &res, void const **inputs, void **outputs) {
            auto a = reinterpret_cast<T const *>(inputs[0]);
            auto b = reinterpret_cast<T const *>(inputs[1]);
            auto y = reinterpret_cast<T *>(outputs[0]);

            if (beta != (T) 0) {
                void const *inputs_[]{inputs[2]};
                void *outputs_[]{outputs[0]};
                (*biasEx)(res, inputs_, outputs_);
            }

            auto handle = res.fetchOrStore<CublasContext>()->handle;
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
            case DataType::F32:
                return lowerTyped<float>(CUDA_R_32F, info, res, biasExpand, strideC0, strideC1);
            case DataType::F64:
                return lowerTyped<double>(CUDA_R_64F, info, res, biasExpand, strideC0, strideC1);
            case DataType::FP16:
                return lowerTyped<half>(CUDA_R_16F, info, res, biasExpand, strideC0, strideC1);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
