#ifndef KERNEL_EXPAND_CNNL_KERNEL_HH
#define KERNEL_EXPAND_CNNL_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ExpandInfoCnnl {
        DataType dataType;
        slice_t<dim_t> inDims, outDims;
    };

    struct ExpandCnnl final : public Kernel {
        ExpandInfoCnnl info;

        explicit ExpandCnnl(ExpandInfoCnnl) noexcept;

        static KernelBox build(Tensor const &input, Tensor const &output) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_EXPAND_CNNL_KERNEL_HH
