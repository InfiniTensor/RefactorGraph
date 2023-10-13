#ifndef KERNEL_CONV_CUDNN_KERNEL_HH
#define KERNEL_CONV_CUDNN_KERNEL_HH

#include "cudnn_impl.h"
#include "kernel/collectors/pool_attributes.hh"

namespace refactor::kernel {

    struct ConvCudnn final : public Kernel {
        cudnn::ConvInfo info;

        ConvCudnn(cudnn::ConvInfo) noexcept;

        static KernelBox build(cudnn::ConvolutionFwdAlgo,
                               PoolAttributes const &,
                               Tensor const &,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId(cudnn::ConvolutionFwdAlgo) noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CUDNN_KERNEL_HH
