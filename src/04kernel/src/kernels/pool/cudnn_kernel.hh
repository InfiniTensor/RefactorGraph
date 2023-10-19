#ifndef KERNEL_POOL_CUDNN_KERNEL_HH
#define KERNEL_POOL_CUDNN_KERNEL_HH

#include "kernel/collectors/pool_attributes.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    /// @brief Use `cudnnPoolingForward`.
    ///        It only supports 4D tensors.
    struct PoolCudnn final : public Kernel {
        struct
        {
        } info;

        explicit PoolCudnn(decltype(info)) noexcept;

        static KernelBox build(PoolType,
                               bool,
                               KernelShape const &,
                               PoolAttributes const &,
                               Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_CUDNN_KERNEL_HH
