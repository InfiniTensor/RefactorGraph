#ifndef KERNEL_MOE_INFO_H
#define KERNEL_MOE_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct AssignPosInfo {
        uint32_t top, expert_num;
        uint32_t elementSize;
        
        AssignPosInfo(uint32_t top, uint32_t expert_num, Tensor const &gate);        
    };

    struct ReorderInfo{
        bool scatter;  
        uint32_t top;
        uint32_t blockNum, blockSize;
        ReorderInfo(bool scatter, uint32_t top, TensorRefs inputs);
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
