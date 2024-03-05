#ifndef KERNEL_POOL_CNNL_KERNEL_HH
#define KERNEL_POOL_CNNL_KERNEL_HH

#include "kernel/attributes/pool_attributes.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    /// @brief Use `cnnlPoolingForward`.
    ///        It only supports 4D tensors.
    struct PoolCnnl final : public Kernel {
        struct
        {
            PoolType poolType;
            DataType dt;
            int xShape[4],
                yShape[4],
                kernelShape[2],
                pads[4],
                strides[2],
                dilations[2];
            bool ceil;
        } info;

        explicit PoolCnnl(decltype(info)) noexcept;

        static KernelBox build(PoolType,
                               bool,
                               KernelShape const &,
                               PoolAttributes const &,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_CNNL_KERNEL_HH
