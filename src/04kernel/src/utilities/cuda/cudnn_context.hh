#ifndef KERNEL_CUDNN_CONTEXT_HH
#define KERNEL_CUDNN_CONTEXT_HH

#include "runtime/resource.h"
#include <cudnn.h>

namespace refactor::kernel::cudnn {

    struct CudnnContext final : public runtime::Resource {
        cudnnHandle_t handle;

        CudnnContext();
        ~CudnnContext();
        CudnnContext(CudnnContext const &) noexcept = delete;
        CudnnContext(CudnnContext &&) noexcept = delete;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build();

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_CONTEXT_HH
