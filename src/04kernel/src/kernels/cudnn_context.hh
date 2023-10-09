#ifndef KERNEL_CUDNN_CONTEXT_HH
#define KERNEL_CUDNN_CONTEXT_HH

#include "runtime/resource.h"

namespace refactor::kernel::cuda {

    class CudnnContext : public runtime::Resource {
        class __Implement;
        __Implement *_impl;

    public:
        CudnnContext() noexcept;
        ~CudnnContext() noexcept;
        CudnnContext(CudnnContext const &) noexcept = delete;
        CudnnContext(CudnnContext &&) noexcept = delete;

        static size_t typeId() noexcept;

        size_t resourceTypeId() const noexcept final = 0;
        std::string_view description() const noexcept final = 0;
    };

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDNN_CONTEXT_HH
