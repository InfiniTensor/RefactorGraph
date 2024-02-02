#ifndef KERNEL_ATTENTION_INFO_H
#define KERNEL_ATTENTION_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct AttentionInfo {
        DataType dataType;
        dim_t batch, nHead, nKVHead, seqLen, headDim, cacheLen;
        bool concatCache, resetCache;
    };

}// namespace refactor::kernel

#endif// KERNEL_ATTENTION_INFO_H
