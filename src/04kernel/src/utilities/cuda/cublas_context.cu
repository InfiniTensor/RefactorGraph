#include "common.h"
#include "cublas_context.hh"

namespace refactor::kernel::cublas {

    CublasContext::CublasContext() : runtime::Resource() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            RUNTIME_ERROR("Failed to create cublas handle");
        }
    }
    CublasContext::~CublasContext() noexcept(false) {
        if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
            RUNTIME_ERROR("Failed to destroy cublas handle");
        }
    }

    auto CublasContext::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto CublasContext::build() noexcept -> runtime::ResourceBox {
        return std::make_unique<CublasContext>();
    }

    auto CublasContext::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto CublasContext::description() const noexcept -> std::string_view {
        return "CublasContext";
    }

}// namespace refactor::kernel::cublas
