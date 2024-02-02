#ifndef KERNEL_CUDA_REDUCE_CUH
#define KERNEL_CUDA_REDUCE_CUH

#include <cub/warp/warp_reduce.cuh>

namespace refactor::kernel::cuda {

    template<class T, class ReductionOp>
    __inline__ __device__ T blockReduce(T x, T init, ReductionOp op) {
        using WarpReduce = cub::WarpReduce<T>;
        __shared__ typename WarpReduce::TempStorage tempStorage;
        __shared__ T shared[32], ans;

        auto reduce = WarpReduce(tempStorage);
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        x = reduce.Reduce(x, op);
        if (lane == 0) { shared[wid] = x; }
        __syncthreads();
        if (wid == 0) {
            x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : init;
            shared[lane] = reduce.Reduce(x, op);
            if (lane == 0) { ans = shared[0]; }
        }
        __syncthreads();
        return ans;// avoid RAW hazard
    }

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_REDUCE_CUH
