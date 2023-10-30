#ifndef KERNEL_CONV_CUDNN_KERNEL_HH
#define KERNEL_CONV_CUDNN_KERNEL_HH

#include "kernel/attributes/pool_attributes.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    namespace cudnn {
        enum class ConvolutionFwdAlgo {
            IMPLICIT_GEMM = 0,
            IMPLICIT_PRECOMP_GEMM = 1,
            GEMM = 2,
            DIRECT = 3,
            FFT = 4,
            FFT_TILING = 5,
            WINOGRAD = 6,
            WINOGRAD_NONFUSED = 7,
            COUNT = 8,
        };
    }// namespace cudnn

    /// @brief Use `cudnnConvolutionForward`.
    ///        It only supports 4D tensors.
    struct ConvCudnn final : public Kernel {
        struct {
            DataType dt;
            cudnn::ConvolutionFwdAlgo algo;
            int xShape[4],
                wShape[4],
                yShape[4],
                dilation[2],
                pad[2],
                stride[2];
        } info;

        explicit ConvCudnn(decltype(info)) noexcept;

        static KernelBox build(cudnn::ConvolutionFwdAlgo,
                               PoolAttributes const &,
                               Tensor const &,
                               Tensor const &,
                               Tensor const &) noexcept;
        static size_t typeId(cudnn::ConvolutionFwdAlgo) noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CUDNN_KERNEL_HH
