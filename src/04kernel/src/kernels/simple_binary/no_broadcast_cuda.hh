#ifndef KERNEL_BINARY_NO_BROADCAST_CUDA_HH
#define KERNEL_BINARY_NO_BROADCAST_CUDA_HH

#include "kernel/collectors/simple_binary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    /// @brief Binary kernel without broadcast support. 11 for 1-1 mapping.
    struct Binary11Cuda final : public Kernel {
        DataType dataType;
        SimpleBinaryType opType;
        size_t size;

        Binary11Cuda(SimpleBinaryType, DataType, size_t) noexcept;

        static KernelBox build(SimpleBinaryType, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_BINARY_NO_BROADCAST_CUDA_HH
