#include "../../utilities/cuda/cublas_context.hh"
#include "../expand/cuda_kernel.hh"
#include "cublas_kernel.hh"

namespace refactor::kernel {
    using namespace runtime;
    using namespace cublas;

    template<class T>
    static auto lowerTyped(cudaDataType_t cudaDataType,
                           MatMulInfo info,
                           Resources &res) noexcept -> RoutineWorkspace {
        // clang-format off
        auto alpha   = static_cast<T>(info.alpha),
             beta    = static_cast<T>(info.biasExpand ? info.beta : 0.0f);
        auto tA      = info.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
             tB      = info.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto m       = info.m,
             n       = info.n,
             k       = info.k;
        auto strideY = m * n,
             strideA = m * k,
             strideB = k * n;
        auto lda     = info.transA ? m : k,
             ldb     = info.transB ? k : n;
        auto biasEx  = info.biasExpand
                     ? std::make_optional(ExpandCuda(*info.biasExpand).lower(res).routine)
                     : std::nullopt;
        // clang-format on
        if (info.broadcaster.needBroadcast()) {
            return [broadcaster = info.broadcaster,
                    cudaDataType,
                    alpha, beta, tA, tB,
                    m, n, k,
                    strideY, strideA, strideB,
                    lda, ldb,
                    biasEx]//
                (Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                    if (biasEx) { (*biasEx)(res, workspace, inputs + 2, outputs); }

                    auto handle = res.fetchOrStore<CublasContext>()->handle;
                    auto a = reinterpret_cast<T const *>(inputs[0]);
                    auto b = reinterpret_cast<T const *>(inputs[1]);
                    auto y = reinterpret_cast<T *>(outputs[0]);
                    uint32_t offset[2];
                    for (auto i : range0_(broadcaster.outputsCount)) {
                        broadcaster.locate(i, offset);
                        CUBLAS_ASSERT(cublasGemmEx(
                            handle,
                            tB, tA, n, m, k,
                            &alpha,
                            b + strideB * offset[1], cudaDataType, ldb,
                            a + strideA * offset[0], cudaDataType, lda,
                            &beta, y + strideY * i, cudaDataType, n,
                            cudaDataType, CUBLAS_GEMM_DEFAULT));
                    }
                };

        } else {
            return [batch = info.broadcaster.outputsCount,
                    cudaDataType,
                    alpha, beta, tA, tB,
                    m, n, k,
                    strideY, strideA, strideB,
                    lda, ldb,
                    biasEx]//
                (Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                    // Call expand kernel to broadcast bias if bias is used
                    if (biasEx) { (*biasEx)(res, workspace, inputs + 2, outputs); }

                    auto handle = res.fetchOrStore<CublasContext>()->handle;
                    auto a = reinterpret_cast<T const *>(inputs[0]);
                    auto b = reinterpret_cast<T const *>(inputs[1]);
                    auto y = reinterpret_cast<T *>(outputs[0]);
                    CUBLAS_ASSERT(cublasGemmStridedBatchedEx(
                        handle,
                        tB, tA, n, m, k,
                        &alpha,
                        b, cudaDataType, ldb, strideB,
                        a, cudaDataType, lda, strideA,
                        &beta, y, cudaDataType, n,
                        strideY, batch,
                        cudaDataType, CUBLAS_GEMM_DEFAULT));
                };
        }
    }

    auto MatMulCublas::lower(Resources &res) const noexcept -> RoutineWorkspace {
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
