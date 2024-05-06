#ifndef KERNEL_TOPK_INFO_H
#define KERNEL_TOPK_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct TopKInfo {
        struct Stride{
            dim_t axis,  in_pre, out_pre;
         };
         struct Size{
            dim_t axis, except_axis;
         };
        uint32_t topk;
        Stride stride;
        Size size;
        
        TopKInfo(uint32_t topk, uint32_t axis, Tensor const &input);     
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
