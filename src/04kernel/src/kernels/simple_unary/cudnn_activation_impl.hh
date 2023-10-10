#ifndef KERNEL_SIMPLE_UNARY_CUDNN_ACTIVATION_IMPL_H
#define KERNEL_SIMPLE_UNARY_CUDNN_ACTIVATION_IMPL_H

#include "kernel/collectors/simple_unary.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel::cudnn {

    Operation lower(SimpleUnaryType, common::DataType, int) noexcept;

}// namespace refactor::kernel::cudnn

#endif// KERNEL_SIMPLE_UNARY_CUDNN_ACTIVATION_IMPL_H
