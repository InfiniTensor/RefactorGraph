#ifndef KERNEL_TRANSPOSE_CPU_KERNEL_HH
#define KERNEL_TRANSPOSE_CPU_KERNEL_HH

#include "common.h"
#include "kernel/collectors/transpose.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct TransposeCpu final : public Kernel {
        TransposeInfo info;

        explicit TransposeCpu(TransposeInfo) noexcept;

        static KernelBox build(TransposeInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CPU_KERNEL_HH
