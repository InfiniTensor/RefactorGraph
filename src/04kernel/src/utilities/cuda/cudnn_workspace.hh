#ifndef KERNEL_CUDNN_WORKSPACE_HH
#define KERNEL_CUDNN_WORKSPACE_HH

#include "runtime/resource.h"
#include <cstddef>

namespace refactor::kernel::cudnn {

    struct CudnnWorkspace final : public runtime::Resource {
        constexpr static size_t size = 4ul << 30;// 4 GiB
        void *ptr;

        CudnnWorkspace() noexcept;
        ~CudnnWorkspace() noexcept;
        CudnnWorkspace(CudnnWorkspace const &) noexcept = delete;
        CudnnWorkspace(CudnnWorkspace &&) noexcept = delete;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build() noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_WORKSPACE_HH
