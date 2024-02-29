#ifndef KERNEL_CUDA_ROPE_CUH
#define KERNEL_CUDA_ROPE_CUH

namespace refactor::kernel::cuda {
    void launchRoPE(
        void const *input,
        int64_t const *posIDs,
        void *output,
        unsigned int batchSize,
        unsigned int seqLen,
        unsigned int nHeads,
        unsigned int headDim,
        float theta,
        bool useHalf);
}

#endif
