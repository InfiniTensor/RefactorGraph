#include "kernel/attributes/topk_info.h"
#include <numeric>

namespace refactor::kernel {

TopKInfo::TopKInfo(uint8_t topk, uint8_t axis, Tensor const &input):topk(topk), 
            axis(axis),
            in_stride(input.strides()[axis]),
            in_stride_pre_axis(axis == 0 ? 0 : input.strides()[axis - 1]),
            out_stride_pre_axis(in_stride_pre_axis/input.shape[axis]*topk),
            elem_size(input.elementsSize()),
            axis_elem_size(input.shape[axis]){}        

}
