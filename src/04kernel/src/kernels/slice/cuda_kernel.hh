#ifndef KERNEL_SPLIT_CUDA_KERNEL_HH
#define KERNEL_SPLIT_CUDA_KERNEL_HH

#include "kernel/attributes/slice_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct SliceCuda final : public Kernel {
        SliceInfo info;

        explicit SliceCuda(SliceInfo) noexcept;

        static KernelBox build(SliceInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CUDA_KERNEL_HH
