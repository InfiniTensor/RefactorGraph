#ifndef KERNEL_ROPE_CUDA_KERNEL_HH
#define KERNEL_ROPE_CUDA_KERNEL_HH


#include "kernel/attributes/rope_info.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    struct RoPECuda final : public Kernel {
        RoPEInfo info;
        DataType dtype;
        RoPECuda(decltype(info), DataType) noexcept;

        static KernelBox build(decltype(info), Tensor const &x) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };
}// namespace refactor::kernel

#endif
