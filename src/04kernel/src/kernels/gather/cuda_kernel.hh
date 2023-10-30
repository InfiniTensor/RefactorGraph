#ifndef KERNEL_GATHER_CUDA_KERNEL_HH
#define KERNEL_GATHER_CUDA_KERNEL_HH

#include "kernel/attributes/gather_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct GatherCuda final : public Kernel {
        GatherInfo info;

        explicit GatherCuda(GatherInfo) noexcept;

        static KernelBox build(GatherInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CUDA_KERNEL_HH
