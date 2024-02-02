#ifndef KERNEL_CUBLASLT_UTILS_CUH
#define KERNEL_CUBLASLT_UTILS_CUH

#include "common.h"
#include "runtime/resource.h"
#include <cublasLt.h>

#define CUBLASLT_ASSERT(STATUS)                                    \
    if (auto status = (STATUS); status != CUBLAS_STATUS_SUCCESS) { \
        fmt::println("cublasLt failed on \"" #STATUS "\" with {}", \
                     (int) status);                                \
        abort();                                                   \
    }

namespace refactor::kernel::cublas {

    struct CublasLtContext final : public runtime::Resource {
        cublasLtHandle_t handle;

        CublasLtContext();
        ~CublasLtContext();
        CublasLtContext(CublasLtContext const &) noexcept = delete;
        CublasLtContext(CublasLtContext &&) noexcept = delete;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build() noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

    cudaDataType dataTypeConvert(DataType);

    class MatMulDescriptor {
        cublasLtMatmulDesc_t _internal;

    public:
        MatMulDescriptor(cublasComputeType_t, cudaDataType);
        ~MatMulDescriptor();
        MatMulDescriptor(MatMulDescriptor const &) noexcept = delete;
        MatMulDescriptor(MatMulDescriptor &&) noexcept = delete;
        cublasLtMatmulDesc_t get() const noexcept;
    };

    struct MatrixLayout {
        cudaDataType dataType;
        uint64_t rows, cols;
        int64_t majorStride;
        cublasLtOrder_t order;
        int32_t batchCount;
        int64_t batchStride;
    };

    class MatrixDescriptor {
        cublasLtMatrixLayout_t _internal;

    public:
        MatrixDescriptor(MatrixLayout layout);
        ~MatrixDescriptor();
        MatrixDescriptor(MatrixDescriptor const &) noexcept = delete;
        MatrixDescriptor(MatrixDescriptor &&) noexcept = delete;
        cublasLtMatrixLayout_t get() const noexcept;
    };

    std::pair<cublasLtMatmulAlgo_t, size_t>
    tune(cublasLtHandle_t,
         MatMulDescriptor const &,
         MatrixDescriptor const &,
         MatrixDescriptor const &,
         MatrixDescriptor const &);

}// namespace refactor::kernel::cublas

#endif// KERNEL_CUBLASLT_UTILS_CUH
