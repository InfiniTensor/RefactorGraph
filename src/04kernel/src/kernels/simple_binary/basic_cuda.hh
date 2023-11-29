#ifndef KERNEL_BINARY_BASIC_CUDA_HH
#define KERNEL_BINARY_BASIC_CUDA_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/simple_binary.h"

namespace refactor::kernel {

    struct BinaryBasicCuda final : public Kernel {
        DataType dataType;
        SimpleBinaryType opType;
        Broadcaster broadcaster;

        BinaryBasicCuda(SimpleBinaryType, DataType, Broadcaster) noexcept;

        static KernelBox build(SimpleBinaryType,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_BINARY_BASIC_CUDA_HH
