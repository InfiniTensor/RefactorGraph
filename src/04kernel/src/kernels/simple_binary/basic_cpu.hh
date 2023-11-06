#ifndef KERNEL_BINARY_BASIC_CPU_HH
#define KERNEL_BINARY_BASIC_CPU_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/simple_binary.h"

namespace refactor::kernel {

    struct BinaryBasicCpu final : public Kernel {
        DataType dataType;
        SimpleBinaryType opType;
        Broadcaster broadcaster;

        BinaryBasicCpu(SimpleBinaryType, DataType, Broadcaster) noexcept;

        static KernelBox build(SimpleBinaryType,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BINARY_BASIC_CPU_HH
