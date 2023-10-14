#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H
#define KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H

#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include "refactor/common.h"

namespace refactor::kernel::cudnn {

    struct BNInfo {
        float epsilon;
        DataType dtX, dtParam;
        LayoutType layout;
        int dimAx[4];// dimA for x, cudnn naming convension

        Routine lower() const __attribute__((weak));
    };

}// namespace refactor::kernel::cudnn

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H
