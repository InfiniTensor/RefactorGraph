#include "cuda_kernel.hh"
#include <cub/cub.cuh>

namespace refactor::kernel {
    using namespace runtime;

    template<class T>
    struct MD {// update the global max and sum, store the output at
               // max and sum
        T max; // store max
        T sum; // store sum
    };
    template<class T>
    __device__ __forceinline__ MD<T> reduce_md_op(MD<T> a, MD<T> b) {
        if (a.max > b.max) {
            return {a.max, a.sum + b.sum * __expf(b.max - a.max)};
        } else {
            return {b.max, b.sum + a.sum * __expf(a.max - b.max)};
        }
    }

    template<int BLOCK_DIM, class T>
    __launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
        T const *__restrict input, T *__restrict output, int size, int dimsize, int stride) {
        // if set axis = 1, inputShape=[I,J,K,S]
        // tid = i(JKS) + j(KS) + k(S) + s

        // blockDim.x = size/dimsize = IKS
        // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

        // now, tid = i(JKS) + k(S) + s;
        int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize;

        MD<T> mdPartial{-__FLT_MAX__, 0};
        for (int i = threadIdx.x; i < dimsize; i += BLOCK_DIM) {
            mdPartial = reduce_md_op(mdPartial, {input[tid + i * stride], 1});// reduce the data to one block
        }
        using BlockReduce = cub::BlockReduce<MD<T>, BLOCK_DIM>;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        __shared__ MD<T> mdTotal;
        MD<T> mdBlock = BlockReduce(tempStorage).Reduce(mdPartial, reduce_md_op<T>);
        if (threadIdx.x == 0) {
            mdTotal = mdBlock;// must set threadIdx.x = 0 write the output to memory
        }
        __syncthreads();

        for (int i = threadIdx.x; i < dimsize; i += BLOCK_DIM) {
            auto j = tid + i * stride;
            output[j] = __expf(input[j] - mdTotal.max) * __fdividef(1, mdTotal.sum);
        }
    }

    template<class T> struct SumOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return a + b;
        }
    };
    template<class T> struct MaxOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return max(a, b);
        }
    };
    template<class ReductionOp, class T>
    __device__ __forceinline__ T WarpAllReduce(T val, int threadGroupWidth, ReductionOp op) {
        for (int mask = threadGroupWidth / 2; mask > 0; mask /= 2) {
            val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }

    template<class T>
    __global__ void _warpSoftmaxKernel(
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
                maxData = max(maxData, input[tid + i * stride]);
            }
            maxData = WarpAllReduce(maxData, blockDim.x, MaxOp<T>{});
            if (threadIdx.x == 0) {
                maxTotal[threadIdx.y] = maxData;
            }

            //--------------------------------------------
            T sumData = 0;
            for (int i = threadIdx.x; i < dimsize; i += blockDim.x) {
                sumData += __expf(input[tid + i * stride] - maxTotal[threadIdx.y]);
            }
            sumData = WarpAllReduce(sumData, blockDim.x, SumOp<T>{});
            if (threadIdx.x == 0) {
                sumTotal[threadIdx.y] = sumData;
            }

            //--------------------------------------------
            for (int i = threadIdx.x; i < dimsize; i += blockDim.x) {
                auto j = tid + i * stride;
                output[j] = __expf(input[j] - maxTotal[threadIdx.y]) * __fdividef(1, sumTotal[threadIdx.y]);
            }
        }
    }

    template<class T>
    Routine lowerTypedCuda(SoftmaxInfo info) {
        using namespace runtime;

        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<T const *>(inputs[0]);
            auto y = reinterpret_cast<T *>(outputs[0]);
            int numBlocks = info.pre * info.post,
                dimsize = info.mid,
                size = numBlocks * dimsize,
                stride = info.post;
            if (dimsize > 1024) {
                _blockSoftmaxKernel<1024><<<numBlocks, 1024>>>(x, y, size, dimsize, stride);
            } else {
                // clang-format off
                int blockDimX = dimsize > 31 ? 32
                              : dimsize > 15 ? 16
                              : dimsize >  7 ?  8
                                             :  4,
                    blockDimY = 1024 / blockDimX;
                // clang-format on
                _warpSoftmaxKernel<<<(numBlocks + blockDimY - 1) / blockDimY,
                                     dim3(blockDimX, blockDimY),
                                     blockDimY * 2 * sizeof(T)>>>(x, y, size, dimsize, stride);
            }
        };
    }

    auto SoftmaxCuda::lower(Resources &res) const noexcept -> RoutineWorkspace {
        switch (info.type.internal) {
            case DataType::F32:
                return lowerTypedCuda<float>(info);
            case DataType::F64:
                return lowerTypedCuda<double>(info);
            // case DataType::FP16:
            //     return lowerTypedCuda<half>(info);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
