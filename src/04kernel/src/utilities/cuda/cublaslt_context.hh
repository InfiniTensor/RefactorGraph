#ifndef KERNEL_CUBLASLT_CONTEXT_HH
#define KERNEL_CUBLASLT_CONTEXT_HH

#include "runtime/resource.h"
#include <cublasLt.h>

#define CUBLAS_ASSERT(STATUS)                                      \
    if (auto status = (STATUS); status != CUBLAS_STATUS_SUCCESS) { \
        fmt::println("cublas failed on \"" #STATUS "\" with {}",   \
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

}// namespace refactor::kernel::cublas

#endif// KERNEL_CUBLASLT_CONTEXT_HH
