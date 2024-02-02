#ifndef KERNEL_ATTENTION_CUDA_KERNEL_HH
#define KERNEL_ATTENTION_CUDA_KERNEL_HH

#include "kernel/attributes/attention_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct AttentionCuda final : public Kernel {
        AttentionInfo info;

        AttentionCuda(decltype(info)) noexcept;

        static KernelBox build(decltype(info)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_ATTENTION_CUDA_KERNEL_HH
