#ifndef KERNEL_CONV_CUDNN_IMPL_H
#define KERNEL_CONV_CUDNN_IMPL_H

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel::cudnn {

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

    struct ConvInfo {
        DataType dt;
        ConvolutionFwdAlgo algo;
        int xShape[4],
            wShape[4],
            yShape[4],
            dilation[2],
            pad[2],
            stride[2];

        Routine lower() const;
    };

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CONV_CUDNN_IMPL_H
