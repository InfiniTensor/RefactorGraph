#ifndef KERNEL_WHERE_CPU_HH
#define KERNEL_WHERE_CPU_HH

#include "kernel/attributes/where_info.h"
#include "kernel/collectors/where.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct WhereCpu final : public Kernel {
        DataType dataType;
        WhereBroadcast broadcast;

        WhereCpu(DataType, WhereBroadcast) noexcept;

        static KernelBox build(Tensor const &, Tensor const &, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };
}// namespace refactor::kernel

#endif// KERNEL_WHERE_CPU_HH
