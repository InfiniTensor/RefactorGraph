#ifndef KERNEL_CAST_CNNL_KERNEL_HH
#define KERNEL_CAST_CNNL_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct CastCnnl final : public Kernel {
        DataType from, to;
        std::vector<int> shape;

        CastCnnl(decltype(from), decltype(to), decltype(shape)) noexcept;

        static KernelBox build(Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CAST_CNNL_KERNEL_HH
