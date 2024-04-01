#ifndef KERNEL_MOE_INFO_H
#define KERNEL_MOE_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct AssignPosInfo {
        int64_t top, expert_num;
        int64_t elementSize;
        
        AssignPosInfo(int64_t top, int64_t expert_num, Tensor const &gate);        
    };

    struct ReorderInfo{
        bool scatter;  
        int64_t top;
        int64_t blockNum, blockSize;
        ReorderInfo(bool scatter, int64_t top, TensorRefs inputs);
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
