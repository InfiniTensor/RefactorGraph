#include "cuda_kernel.hh"
#include <cub/cub.cuh>

namespace refactor::kernel {
    using K = SoftmaxCuda;
    using DT = DataType;
    using namespace runtime;


    template<typename T>
    struct MD {   // update the global max and sum, store the output at
                  // max_tmp and sum_tmp
        T max_tmp;// store max
        T sum_tmp;// store sum
    };
    template<typename T>
    __device__ __forceinline__ MD<T> reduce_md_op(MD<T> a, MD<T> b) {
        bool a_bigger = (a.max_tmp > b.max_tmp);
        auto bigger = a_bigger ? a : b;
        auto smaller = a_bigger ? b : a;
        return {bigger.max_tmp, bigger.sum_tmp + smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp)};
    }

    template<int BLOCK_DIM, typename T>
    __launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
        T const *__restrict input, T *__restrict output, int size, int dimsize,
        int stride) {// if set axis = 1, inputShape=[I,J,K,S]
                     // tid = i(JKS) + j(KS) + k(S) + s

        // blockDim.x = size/dimsize = IKS
        // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

        int tid =
            blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                      dimsize;// now, tid = i(JKS) + k(S) + s;

        MD<T> md_partial;
        md_partial.max_tmp = -__FLT_MAX__;
        md_partial.sum_tmp = 0.0f;
        MD<T> md_input;
        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

            md_input.max_tmp = input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];

            md_input.sum_tmp = 1.0f;
            md_partial = reduce_md_op(md_partial,
                                      md_input);// reduce the data to one block
        }
        typedef cub::BlockReduce<MD<T>, BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ MD<T> md_total;
        MD<T> md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op<T>);
        if (threadIdx.x ==
            0) {// must set threadIdx.x = 0 write the output to memory
            md_total = md_block;
        }
        __syncthreads();

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
            output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                       md_total.max_tmp) *
                __fdividef(1.0F, md_total.sum_tmp);
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
             int thread_group_width>
    __inline__ __device__ T WarpAllReduce(T val) {
        for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
            val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }
#define max_function(a, b) ((a) > (b) ? (a) : (b))
    template<int BLOCK_DIM_x, int BLOCK_DIM_y, typename T>
    __global__ void _warpSoftmaxKernel(T const *__restrict input,
                                       T *__restrict output, int size,
                                       int dimsize, int stride) {
        int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
        int otherSize = size / dimsize;
        int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

        if (otherIdx < otherSize) {

            __shared__ float max_total[BLOCK_DIM_y];
            __shared__ float sum_total[BLOCK_DIM_y];
            T max_data = -__FLT_MAX__;

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
                max_data = max_function(
                    max_data,
                    input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride]);
            }

            max_data = WarpAllReduce<MaxOp, T, BLOCK_DIM_x>(max_data);

            if (threadIdx.x == 0)
                max_total[threadIdx.y] = max_data;

            //--------------------------------------------
            T sum_data = 0.0f;

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
                sum_data +=
                    __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                           max_total[threadIdx.y]);
            }

            sum_data = WarpAllReduce<SumOp, T, BLOCK_DIM_x>(sum_data);

            if (threadIdx.x == 0)
                sum_total[threadIdx.y] = sum_data;

            //--------------------------------------------

            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
                output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                    __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                           max_total[threadIdx.y]) *
                    __fdividef(1.0F, sum_total[threadIdx.y]);
            }
        }
    }

    template<decltype(DataType::internal) T>
    Routine lowerTypedCuda(SoftmaxInfo info) {
        using namespace runtime;
        using dt = typename primitive<T>::type;

        return [info](Resources &, void const **inputs, void **outputs) {
            auto x = reinterpret_cast<dt const *>(inputs[0]);
            auto y = reinterpret_cast<dt *>(outputs[0]);
            int num_blocks = info.pre * info.post;
            int dimsize = info.mid;
            int size = info.size;
            int stride = info.post;
            if (dimsize > 1024) {
                _blockSoftmaxKernel<1024><<<num_blocks, 1024>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 31) {
                int BLOCK_DIM_x = 32;
                int BLOCK_DIM_y = 32;
                int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);
                _warpSoftmaxKernel<32, 32>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 15) {
                int BLOCK_DIM_x = 16;
                int BLOCK_DIM_y = 64;
                int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);

                _warpSoftmaxKernel<16, 64>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else if (dimsize > 7) {
                int BLOCK_DIM_x = 8;
                int BLOCK_DIM_y = 128;
                int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);

                _warpSoftmaxKernel<8, 128>
                    <<<grid_dim, block_dim>>>(x, y, size, dimsize, stride);
            } else {
                int BLOCK_DIM_x = 4;
                int BLOCK_DIM_y = 256;
                int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);

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
