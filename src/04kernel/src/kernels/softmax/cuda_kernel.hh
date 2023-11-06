#ifndef KERNEL_SOFTMAX_CUDA_HH
#define KERNEL_SOFTMAX_CUDA_HH

#include "cpu_kernel.hh"
#include "kernel/attributes/softmax_info.h"
#include "kernel/collectors/softmax.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SoftmaxCuda final : public Kernel {
        SoftmaxInfo info;

        SoftmaxCuda(SoftmaxInfo) noexcept;
        static KernelBox build(SoftmaxInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &res) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif//KERNEL_SOFTMAX_CUDA_HH
