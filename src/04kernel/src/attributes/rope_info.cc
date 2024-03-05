#include "kernel/attributes/rope_info.h"

namespace refactor::kernel {
    RoPEInfo::RoPEInfo(Tensor const &input, float _theta) : theta(_theta) {
        if (input.rank() == 4) {
            batchsize = input.shape[0];
            seq_len = input.shape[1];
            n_heads = input.shape[2];
            head_dim = input.shape[3];
        } else {
            batchsize = 1;
            seq_len = input.shape[0];
            n_heads = input.shape[1];
            head_dim = input.shape[2];
        }
    }
}// namespace refactor::kernel
