#ifndef KERNEL_PAD_CUDA_HH
#define KERNEL_PAD_CUDA_HH

#include "kernel/attributes/pad_info.h"
#include "kernel/collectors/pad.h"

namespace refactor::kernel {

    struct PadCuda final : public Kernel {
        PadInfo info;

        PadCuda(PadInfo) noexcept;
        static KernelBox build(PadInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif//KERNEL_PAD_CUDA_HH
