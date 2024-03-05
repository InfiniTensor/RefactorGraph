#ifndef KERNEL_ROPE_INFO_H
#define KERNEL_ROPE_INFO_H

#include "../tensor.h"

namespace refactor::kernel {
    struct RoPEInfo {
        dim_t batchsize = 1;
        dim_t seq_len, n_heads, head_dim;
        float theta;

        RoPEInfo(Tensor const &input, float _theta);
    };

}// namespace refactor::kernel

#endif
