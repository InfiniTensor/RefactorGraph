#ifndef KERNEL_GATHER_CPU_KERNEL_HH
#define KERNEL_GATHER_CPU_KERNEL_HH
#include "gather_helper.hh"
#include "kernel/collectors/gather.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct GatherCpu final : public Kernel {
        DataType indexType;
        GatherMetaData metaData;

        explicit GatherCpu(DataType, GatherMetaData) noexcept;

        static KernelBox build(Tensor const &, Tensor const &, Tensor const &, uint32_t) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif
