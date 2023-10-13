#ifndef KERNEL_ARTHIMETIC11_CUDA_HH
#define KERNEL_ARTHIMETIC11_CUDA_HH

#include "common/data_type.h"
#include "kernel/collectors/simple_binary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct Arthimetic11Cuda final : public Kernel {
        common::DataType dataType;
        SimpleBinaryType opType;
        size_t size;

        Arthimetic11Cuda(SimpleBinaryType, common::DataType, size_t) noexcept;

        static KernelBox build(SimpleBinaryType, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_ARTHIMETIC11_CUDA_HH
