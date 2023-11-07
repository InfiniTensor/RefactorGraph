#ifndef KERNEL_SOFTMAX_CPU_KERNEL_HH
#define KERNEL_SOFTMAX_CPU_KERNEL_HH

#include "kernel/attributes/softmax_info.h"
#include "kernel/collectors/softmax.h"

namespace refactor::kernel {

    struct SoftmaxCpu final : public Kernel {
        SoftmaxInfo info;

        explicit SoftmaxCpu(SoftmaxInfo) noexcept;

        static KernelBox build(SoftmaxInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_CPU_KERNEL_HH
