#ifndef KERNEL_BINARY_BASIC_CPU_HH
#define KERNEL_BINARY_BASIC_CPU_HH

#include "common.h"
#include "kernel/collectors/broadcaster.hpp"
#include "kernel/collectors/simple_binary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    class BinaryBroadcast {
        std::vector<uint_lv2> _strides;
        uint_lv2 _size;

    public:
        BinaryBroadcast(Shape const &, Shape const &) noexcept;
        std::pair<uint_lv2, uint_lv2> locate(uint_lv2) const noexcept;
        uint_lv2 size() const noexcept;
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
