#ifndef KERNEL_BINARY_BASIC_CPU_HH
#define KERNEL_BINARY_BASIC_CPU_HH

#include "common.h"
#include "kernel/collectors/broadcaster.hpp"
#include "kernel/collectors/simple_binary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct BinaryBroadcast {
        union Dimension {
            struct {
                uint32_t
                    size : 30;
                bool
                    a : 1,
                    b : 1;
            };
            uint32_t code;
        };
        std::vector<Dimension> dims;

        BinaryBroadcast(Shape const &, Shape const &) noexcept;
        std::pair<uint32_t, uint32_t> locate(uint32_t) const noexcept;
        uint32_t size() const noexcept;

    private:
        Dimension push(Dimension next, Dimension state, uint32_t size) noexcept;
    };

    struct BinaryBasicCpu final : public Kernel {
        DataType dataType;
        SimpleBinaryType opType;
        BinaryBroadcast broadcast;

        BinaryBasicCpu(SimpleBinaryType, DataType, BinaryBroadcast) noexcept;

        static KernelBox build(SimpleBinaryType,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BINARY_BASIC_CPU_HH
