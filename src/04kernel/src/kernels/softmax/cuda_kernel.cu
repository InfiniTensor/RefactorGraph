#include "cuda_kernel.hh"
#include "kernel/cuda/reduce.cuh"

namespace refactor::kernel {
    using namespace runtime;

    template<class T>
    __device__ __forceinline__ T exp_(T x);
    template<> __device__ __forceinline__ float exp_<float>(float x) { return expf(x); }
    template<> __device__ __forceinline__ double exp_<double>(double x) { return exp(x); }
    template<> __device__ __forceinline__ half exp_<half>(half x) { return hexp(x); }
    template<> __device__ __forceinline__ nv_bfloat16 exp_<nv_bfloat16>(nv_bfloat16 x) { return hexp(x); }

    template<class T> __device__ __forceinline__ T reciprocal(T x);
    template<> __device__ __forceinline__ float reciprocal<float>(float x) { return fdividef(1, x); }
    template<> __device__ __forceinline__ double reciprocal<double>(double x) { return 1 / x; }
    template<> __device__ __forceinline__ half reciprocal<half>(half x) { return hrcp(x); }
    template<> __device__ __forceinline__ nv_bfloat16 reciprocal<nv_bfloat16>(nv_bfloat16 x) { return hrcp(x); }

    // blockDim.x === BLOCK_DIM
    template<class T>
    __global__ void blockSoftmaxKernel(
        T const *__restrict x,
        T *__restrict y,
        int mid,
        int stride) {
        int id = (blockIdx.x - blockIdx.x % stride) * mid + blockIdx.x % stride;

        struct MaxSum {
            T max, sum;

            static __device__ __forceinline__ MaxSum reduce(MaxSum a, MaxSum b) {
                if (a.max > b.max) {
                    return {a.max, a.sum + b.sum * exp_(b.max - a.max)};
                } else {
                    return {b.max, b.sum + a.sum * exp_(a.max - b.max)};
                }
            }
        } maxSumThread{x[id], 1};
        for (int i = threadIdx.x + blockDim.x; i < mid; i += blockDim.x) {
            maxSumThread = MaxSum::reduce(maxSumThread, {x[id + i * stride], 1});// reduce the data to one block
        }
        __shared__ MaxSum maxSumTotal;
        auto maxSumBlock = cuda::blockReduce(maxSumThread, {-__FLT_MAX__, 0}, MaxSum::reduce);
        if (threadIdx.x == 0) {
            maxSumTotal = maxSumBlock;// must set threadIdx.x = 0 write the output to memory
        }
        __syncthreads();

        for (int i = threadIdx.x; i < mid; i += blockDim.x) {
            auto j = id + i * stride;
            y[j] = exp_(x[j] - maxSumTotal.max) * reciprocal(maxSumTotal.sum);
        }
    }

    template<class T, class ReductionOp>
    __device__ __forceinline__ T WarpAllReduce(T val, ReductionOp op) {
        for (int mask = blockDim.x >> 1; mask > 0; mask >>= 1) {
            val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }

    template<class T>
    __global__ void warpSoftmaxKernel(
        T const *__restrict input,
        T *__restrict output,
        int size, int dimsize, int stride) {

        int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
        int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

        extern __shared__ char shared[];
        if (otherIdx < size / dimsize) {
            auto maxTotal = reinterpret_cast<T *>(shared),
                 sumTotal = maxTotal + blockDim.y;

            T maxData = -__FLT_MAX__;
            for (int i = threadIdx.x; i < dimsize; i += blockDim.x) {
                maxData = CUB_MAX(maxData, input[tid + i * stride]);
            }
            maxData = WarpAllReduce(maxData, cub::Max());
            if (threadIdx.x == 0) {
                maxTotal[threadIdx.y] = maxData;
            }

            //--------------------------------------------
            T sumData = 0;
            for (int i = threadIdx.x; i < dimsize; i += blockDim.x) {
                sumData += exp_(input[tid + i * stride] - maxTotal[threadIdx.y]);
            }
            sumData = WarpAllReduce(sumData, cub::Sum());
            if (threadIdx.x == 0) {
                sumTotal[threadIdx.y] = sumData;
            }

            //--------------------------------------------
            for (int i = threadIdx.x; i < dimsize; i += blockDim.x) {
                auto j = tid + i * stride;
                output[j] = exp_(input[j] - maxTotal[threadIdx.y]) * reciprocal(sumTotal[threadIdx.y]);
            }
        }
    }

    template<class T>
    Routine lowerTypedCuda(SoftmaxInfo info) {
        using namespace runtime;

        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<T const *>(inputs[0]);
            auto y = reinterpret_cast<T *>(outputs[0]);
            int numBlocks = info.pre * info.post;
            if (info.mid > 1024) {
                blockSoftmaxKernel<<<numBlocks, 1024>>>(x, y, info.mid, info.post);
            } else {
                int blockDimX, mid = static_cast<int>(info.mid);
                for (blockDimX = 32; blockDimX > 4 && mid < blockDimX; blockDimX /= 2) {}
                auto blockDimY = 1024 / blockDimX;
                warpSoftmaxKernel<<<(numBlocks + blockDimY - 1) / blockDimY,
                                    dim3(blockDimX, blockDimY),
                                    blockDimY * 2 * sizeof(T)>>>(x, y, numBlocks * mid, mid, info.post);
            }
        };
    }

    auto SoftmaxCuda::lower(Resources &res) const noexcept -> RoutineWorkspace {
        switch (info.type.internal) {
            case DataType::F32:
                return lowerTypedCuda<float>(info);
            case DataType::F64:
                return lowerTypedCuda<double>(info);
            case DataType::FP16:
                return lowerTypedCuda<half>(info);
            case DataType::BF16:
                return lowerTypedCuda<nv_bfloat16>(info);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
