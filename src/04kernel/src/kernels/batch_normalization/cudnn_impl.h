#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H
#define KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H

#include "common/data_type.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel::cudnn {

    struct Info {
        float epsilon;
        common::DataType dtX, dtParam;
        LayoutType layout;
        int dimAx[4];// dimA for x, cudnn naming convension

        Operation lower() const;
    };


}// namespace refactor::kernel::cudnn

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_IMPL_H
