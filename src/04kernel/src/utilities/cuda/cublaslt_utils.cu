#include "cublaslt_utils.cuh"
#include "hardware/devices/nvidia.h"

namespace refactor::kernel::cublas {

    CublasLtContext::CublasLtContext() : runtime::Resource() {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            RUNTIME_ERROR("Failed to create cublasLt handle");
        }
    }
    CublasLtContext::~CublasLtContext() {
        if (cublasLtDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
            fmt::println("Failed to destroy cublasLt handle");
            abort();
        }
    }

    auto CublasLtContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto CublasLtContext::build() noexcept -> runtime::ResourceBox {
        return std::make_unique<CublasLtContext>();
    }

    auto CublasLtContext::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto CublasLtContext::description() const noexcept -> std::string_view {
        return "CublasLtContext";
    }

    cudaDataType dataTypeConvert(DataType dt) {
        switch (dt) {
            case DataType::F32:
                return CUDA_R_32F;
            default:
                TODO("");
        }
    }

    MatMulDescriptor::MatMulDescriptor(cublasComputeType_t compute, cudaDataType data)
        : _internal(nullptr) {
        CUBLASLT_ASSERT(cublasLtMatmulDescCreate(&_internal, compute, data));
    }
    MatMulDescriptor::~MatMulDescriptor() {
        CUBLASLT_ASSERT(cublasLtMatmulDescDestroy(_internal));
    }
    cublasLtMatmulDesc_t MatMulDescriptor::get() const noexcept {
        return _internal;
    }

    MatrixDescriptor::MatrixDescriptor(MatrixLayout layout)
        : _internal(nullptr) {
        CUBLASLT_ASSERT(cublasLtMatrixLayoutCreate(
            &_internal,
            layout.dataType,
            layout.rows,
            layout.cols,
            layout.majorStride));
        CUBLASLT_ASSERT(cublasLtMatrixLayoutSetAttribute(
            _internal,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &layout.order,
            sizeof(layout.order)));
        CUBLASLT_ASSERT(cublasLtMatrixLayoutSetAttribute(
            _internal,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &layout.batchCount,
            sizeof(layout.batchCount)));
        CUBLASLT_ASSERT(cublasLtMatrixLayoutSetAttribute(
            _internal,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &layout.batchStride,
            sizeof(layout.batchStride)));
    }
    MatrixDescriptor::~MatrixDescriptor() {
        CUBLASLT_ASSERT(cublasLtMatrixLayoutDestroy(_internal));
    }
    cublasLtMatrixLayout_t MatrixDescriptor::get() const noexcept {
        return _internal;
    }

    std::pair<cublasLtMatmulAlgo_t, size_t>
    tune(cublasLtHandle_t handle,
         MatMulDescriptor const &matmul,
         MatrixDescriptor const &a,
         MatrixDescriptor const &b,
         MatrixDescriptor const &c) {

        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));

        auto workspace = std::numeric_limits<uint64_t>::max();
        auto alignment = prop.textureAlignment;

        cublasLtMatmulPreference_t preference;
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceCreate(&preference));
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace,
            sizeof(workspace)));
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &alignment,
            sizeof(alignment)));
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            &alignment,
            sizeof(alignment)));
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            &alignment,
            sizeof(alignment)));
        CUBLASLT_ASSERT(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            &alignment,
            sizeof(alignment)));

        cublasLtMatmulHeuristicResult_t result;
        int ansN;
        CUBLASLT_ASSERT(cublasLtMatmulAlgoGetHeuristic(
            handle,
            matmul.get(),
            a.get(),
            b.get(),
            c.get(),
            c.get(),
            preference,
            1,
            &result,
            &ansN));
        ASSERT(ansN == 1, "");

        return {result.algo, result.workspaceSize};
    }

}// namespace refactor::kernel::cublas
