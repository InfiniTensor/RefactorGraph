#ifndef KERNEL_PAD_2D_INFO_H
#define KERNEL_PAD_2D_INFO_H

#include "kernel/tensor.h"

namespace refactor::kernel {
    /// @brief 优化用于计算的 Slice 描述。
    struct Pad2DInfo {
        dim_t blockCount, blockSize, hw, w;
        ddim_t padHW, padW;

        Pad2DInfo(DataType, slice_t<dim_t> input, ddim_t const *pads);
    };

}// namespace refactor::kernel

#endif// KERNEL_PAD_2D_INFO_H
