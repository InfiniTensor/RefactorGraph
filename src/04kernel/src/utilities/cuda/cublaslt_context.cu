#include "common.h"
#include "cublaslt_context.hh"

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

}// namespace refactor::kernel::cublas
