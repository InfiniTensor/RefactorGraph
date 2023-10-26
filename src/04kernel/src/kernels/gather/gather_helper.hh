#ifndef KERNEL_GATHER_HELPER_HH
#define KERNEL_GATHER_HELPER_HH

#include "kernel/tensor.h"

namespace refactor::kernel {
    // Info for gather operation used by cpu
    struct GatherMetaData {
        // Size (Bytes) of one input element
        size_t itemSize;
        // Output element count
        size_t outSize;
        // Axis of the gather operation
        uint32_t axis;
        // Rank of input
        uint32_t inNDim;
        // Rank of output
        uint32_t outNDim;
        // Rank of indices
        uint32_t idxNDim;
        // Shape of indices
        Shape idxShape;
        // Shape of output
        Shape outShape;
        // Strides of input
        Strides inStrides;
        // Strides of indices
        Strides idxStrides;
    };

    // Info for gather operation used by cuda kernel
    struct GatherCudaMetaData {
        // Size (Bytes) of one input element
        size_t itemSize;
        // Output element count
        size_t outSize;
        // Axis of the gather operation
        uint32_t axis;
        // Rank of input
        uint32_t inNDim;
        // Rank of output
        uint32_t outNDim;
        // Rank of indices
        uint32_t idxNDim;
        // Shape of indices
        uint32_t const *idxShape;
        // Shape of output
        uint32_t const *outShape;
        // Strides of input
        size_t const *inStrides;
        // Strides of indices
        size_t const *idxStrides;
    };
}// namespace refactor::kernel
#endif
