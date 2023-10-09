#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_H
#define KERNEL_BATCH_NORMALIZATION_CUDNN_H

#include "common/data_type.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel::cudnn {

    Operation lower(
        float epsilon,
        std::array<common::DataType, 3>,
        Shape,
        uint32_t paramSize);

}// namespace refactor::kernel::cudnn

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_H
