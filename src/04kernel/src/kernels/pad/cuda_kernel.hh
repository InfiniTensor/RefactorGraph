#ifndef KERNEL_PAD_CUDA_HH
#define KERNEL_PAD_CUDA_HH

#include "kernel/attributes/pad_info.h"
#include "kernel/collectors/pad.h"

namespace refactor::kernel {

    struct PadCuda final : public Kernel {
        PadInfo info;
        PadType mode;
        size_t valueLength;

        PadCuda(PadInfo, PadType, size_t) noexcept;
        static KernelBox build(PadInfo, PadType, std::optional<std::reference_wrapper<Tensor const>>) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif//KERNEL_PAD_CUDA_HH
