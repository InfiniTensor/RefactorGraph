#ifndef KERNEL_CUBLAS_CONTEXT_HH
#define KERNEL_CUBLAS_CONTEXT_HH

#include "runtime/resource.h"
#include <cublas_v2.h>

namespace refactor::kernel::cublas {

    struct CublasContext final : public runtime::Resource {
        cublasHandle_t handle;

        CublasContext();
        ~CublasContext();
        CublasContext(CublasContext const &) noexcept = delete;
        CublasContext(CublasContext &&) noexcept = delete;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build() noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::kernel::cublas

#endif// KERNEL_CUBLAS_CONTEXT_HH
