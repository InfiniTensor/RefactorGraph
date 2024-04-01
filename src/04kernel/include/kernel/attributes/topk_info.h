#ifndef KERNEL_TOPK_INFO_H
#define KERNEL_TOPK_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct TopKInfo {
        
        int64_t topk;
        int64_t axis;
        size_t in_stride, in_stride_pre_axis, out_stride_pre_axis;
        size_t elem_size, axis_elem_size;
        
        TopKInfo(int64_t topk, int64_t axis, Tensor const &input);
        size_t getElementSize() const {return  elem_size;}
        size_t getAxisElementSize()const { return axis_elem_size;}
        size_t getInStride()const{return in_stride;}
        size_t getInStridePreAxis()const{return in_stride_pre_axis;}
        size_t getOutStridePreAxis()const {return out_stride_pre_axis;}
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
