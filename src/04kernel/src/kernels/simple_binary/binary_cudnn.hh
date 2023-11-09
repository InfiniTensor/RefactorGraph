#ifndef KERNEL_BINARY_CUDNN_HH
#define KERNEL_BINARY_CUDNN_HH

#include "kernel/collectors/simple_binary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct BinaryCudnn final : public Kernel {
        DataType dataType;
        SimpleBinaryType opType;
        std::vector<int> aDims, bDims, cDims;

        BinaryCudnn(SimpleBinaryType, DataType, std::vector<int> aDims_, std::vector<int> bDims_, std::vector<int> cDims_) noexcept;

        static KernelBox build(SimpleBinaryType, Tensor const &, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_BINARY_CUDNN_HH
