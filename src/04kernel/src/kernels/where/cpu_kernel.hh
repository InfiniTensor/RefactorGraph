#ifndef KERNEL_WHERE_CPU_HH
#define KERNEL_WHERE_CPU_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/where.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct WhereCpu final : public Kernel {
        DataType dataType;
        Broadcaster broadcaster;

        WhereCpu(DataType, Broadcaster) noexcept;

        static KernelBox build(TensorRefs const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower(Resources &) const noexcept final;
    };
}// namespace refactor::kernel

#endif// KERNEL_WHERE_CPU_HH
