#ifndef KERNEL_CONCAT_CNNL_KERNEL_HH
#define KERNEL_CONCAT_CNNL_KERNEL_HH

#include "../../kernels/split/cnnl_kernel.hh"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ConcatCnnl final : public Kernel {
        SplitInfoCnnl info;

        explicit ConcatCnnl(SplitInfoCnnl) noexcept;

        static KernelBox build(int, TensorRefs, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CONCAT_CNNL_KERNEL_HH
