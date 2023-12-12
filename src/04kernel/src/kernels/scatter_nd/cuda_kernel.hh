#ifndef KERNEL_SCATTER_ND_CUDA_KERNEL_HH
#define KERNEL_SCATTER_ND_CUDA_KERNEL_HH

#include "kernel/attributes/scatter_nd_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ScatterNDCuda final : public Kernel {
        ScatterNDInfo info;

        explicit ScatterNDCuda(decltype(info));

        static KernelBox build(decltype(info)) noexcept;

        static KernelBox build(Tensor const &, bool hasMax) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SCATTER_ND_CUDA_KERNEL_HH
