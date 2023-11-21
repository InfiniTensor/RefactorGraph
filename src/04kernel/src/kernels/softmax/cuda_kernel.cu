#include "cuda_kernel.hh"
#include <cub/cub.cuh>

namespace refactor::kernel {
    using K = SoftmaxCuda;
    using DT = DataType;
    using namespace runtime;


    template<typename T>
    struct MD {  // update the global max and sum, store the output at
                 // maxTmp and sumTmp
        T maxTmp;// store max
        T sumTmp;// store sum
    };
    template<typename T>
    __device__ __forceinline__ MD<T> reduce_md_op(MD<T> a, MD<T> b) {
        bool compair = (a.maxTmp > b.maxTmp);
        auto bigger = compair ? a : b;
        auto smaller = compair ? b : a;
        return {bigger.maxTmp, bigger.sumTmp + smaller.sumTmp * __expf(smaller.maxTmp - bigger.maxTmp)};
    }

    template<int BLOCK_DIM, typename T>
    __launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
        T const *__restrict input, T *__restrict output, int size, int dimsize,
        int stride) {// if set axis = 1, inputShape=[I,J,K,S]
                     // tid = i(JKS) + j(KS) + k(S) + s

        // blockDim.x = size/dimsize = IKS
        // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

        // now, tid = i(JKS) + k(S) + s;
        int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize;

        MD<T> mdPartial;
        mdPartial.maxTmp = -__FLT_MAX__;
        mdPartial.sumTmp = 0.0f;
        MD<T> mdInput;
        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

            mdInput.maxTmp = input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];

            mdInput.sumTmp = 1.0f;
            mdPartial = reduce_md_op(mdPartial, mdInput);// reduce the data to one block
        }
        typedef cub::BlockReduce<MD<T>, BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        __shared__ MD<T> mdTotal;
        MD<T> mdBlock = BlockReduce(tempStorage).Reduce(mdPartial, reduce_md_op<T>);
        if (threadIdx.x == 0) {
            // must set threadIdx.x = 0 write the output to memory
            mdTotal = mdBlock;
        }
        __syncthreads();

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
            output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                       mdTotal.maxTmp) *
                __fdividef(1.0F, mdTotal.sumTmp);
        }
    }

    template<typename T> struct SumOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return a + b;
        }
    };

    template<typename T> struct MaxOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return max(a, b);
        }
    };
    template<template<typename> class ReductionOp, typename T,
             int threadGroupWidth>
    __inline__ __device__ T WarpAllReduce(T val) {
        for (int mask = threadGroupWidth / 2; mask > 0; mask /= 2) {
            val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }
#define max_function(a, b) ((a) > (b) ? (a) : (b))
    template<int BLOCK_DIM_X, int BLOCK_DIM_Y, typename T>
    __global__ void _warpSoftmaxKernel(T const *__restrict input,
                                       T *__restrict output, int size,
                                       int dimsize, int stride) {
        int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
        int otherSize = size / dimsize;
        int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

        if (otherIdx < otherSize) {

            __shared__ float maxTotal[BLOCK_DIM_Y];
            __shared__ float sumTotal[BLOCK_DIM_Y];
            T maxData = -__FLT_MAX__;

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_X < dimsize; ph++) {
                maxData = max_function(
                    maxData,
                    input[tid + (threadIdx.x + ph * BLOCK_DIM_X) * stride]);
            }

            maxData = WarpAllReduce<MaxOp, T, BLOCK_DIM_X>(maxData);

            if (threadIdx.x == 0)
                maxTotal[threadIdx.y] = maxData;

            //--------------------------------------------
            T sumData = 0.0f;

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_X < dimsize; ph++) {
                sumData +=
                    __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM_X) * stride] -
                           maxTotal[threadIdx.y]);
            }

            sumData = WarpAllReduce<SumOp, T, BLOCK_DIM_X>(sumData);

            if (threadIdx.x == 0)
                sumTotal[threadIdx.y] = sumData;

            //--------------------------------------------

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_X < dimsize; ph++) {
                output[tid + (threadIdx.x + ph * BLOCK_DIM_X) * stride] =
                    __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM_X) * stride] -
                           maxTotal[threadIdx.y]) *
                    __fdividef(1.0F, sumTotal[threadIdx.y]);
            }
        }
    }

    template<decltype(DataType::internal) T>
    Routine lowerTypedCuda(SoftmaxInfo info) {
        using namespace runtime;
        using dt = typename primitive<T>::type;

        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<dt const *>(inputs[0]);
            auto y = reinterpret_cast<dt *>(outputs[0]);
            int numBlocks = info.pre * info.post;
            int dimsize = info.mid;
            int size = numBlocks * dimsize;
            int stride = info.post;
            if (dimsize > 1024) {
                _blockSoftmaxKernel<1024><<<numBlocks, 1024>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 31) {
                int BLOCK_DIM_X = 32;
                int BLOCK_DIM_Y = 32;
                int NUM_BLOCK_X = (numBlocks + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
                dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
                dim3 grid_dim(NUM_BLOCK_X, 1, 1);
                _warpSoftmaxKernel<32, 32>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 15) {
                int BLOCK_DIM_X = 16;
                int BLOCK_DIM_Y = 64;
                int NUM_BLOCK_X = (numBlocks + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
                dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
                dim3 grid_dim(NUM_BLOCK_X, 1, 1);

                _warpSoftmaxKernel<16, 64>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 7) {
                int BLOCK_DIM_X = 8;
                int BLOCK_DIM_Y = 128;
                int NUM_BLOCK_X = (numBlocks + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
                dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
                dim3 grid_dim(NUM_BLOCK_X, 1, 1);

                _warpSoftmaxKernel<8, 128>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else {
                int BLOCK_DIM_X = 4;
                int BLOCK_DIM_Y = 256;
                int NUM_BLOCK_X = (numBlocks + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
                dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
                dim3 grid_dim(NUM_BLOCK_X, 1, 1);

                _warpSoftmaxKernel<4, 256>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            }
        };
    }

    auto K::lower(Resources &res) const noexcept -> Routine {
#define CASE(T) \
    case DT::T: \
        return lowerTypedCuda<DT::T>(info);
        switch (info.type.internal) {
            CASE(F32)
            CASE(F64)
            //CASE(FP16)
            //CASE(BF16)
        }
    }

}// namespace refactor::kernel
