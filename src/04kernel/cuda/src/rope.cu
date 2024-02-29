#include "kernel/cuda/functions.cuh"
#include "kernel/cuda/rope.cuh"
#include <cstdio>
#include <cuda_fp16.h>

namespace refactor::kernel::cuda {

    template<class T>
    __global__ static void RoPEKernel(
        unsigned int dim_model,// dim_model = num_head * dim_head
        unsigned int dim_head,
        unsigned int hidden_stride,     // hidden_stride = seq_len * dim_model
        unsigned int pos_stride,        // pos_stride = seq_len
        const int64_t *__restrict__ pos,// (batch, seq_len)
        const T *__restrict__ in,       // (batch, seq_len, num_head, dim_head)
        T *__restrict__ out             // (batch, seq_len, num_head, dim_head)
    ) {
        unsigned int batch_id = blockIdx.x;
        int64_t target_pos = pos[batch_id * pos_stride + blockIdx.y];
        size_t ith = blockIdx.z * blockDim.x + threadIdx.x;
        unsigned int col = ith % dim_head;
        size_t offset = batch_id * hidden_stride + blockIdx.y * dim_model;

        if (ith >= dim_model)
            return;

        unsigned int half_dim = dim_head / 2;
        if (col < half_dim) {
            float freq = target_pos * powf(10000, -float(col * 2) / dim_head);
            float cos_freq = cos(freq);
            float sin_freq = sin(freq);
            out[offset + ith] =
                in[offset + ith] * T(cos_freq) - in[offset + ith + half_dim] * T(sin_freq);
        } else {
            float freq = target_pos * powf(10000, -float((col - half_dim) * 2) / dim_head);
            float cos_freq = cos(freq);
            float sin_freq = sin(freq);
            out[offset + ith] =
                in[offset + ith] * T(cos_freq) + in[offset + ith - half_dim] * T(sin_freq);
        }
    }


    void launchRoPE(
        void const *input,
        int64_t const *posIDs,
        void *output,
        unsigned int batchSize,
        unsigned int seqLen,
        unsigned int nHeads,
        unsigned int headDim,
        float theta,
        bool useHalf) {

        unsigned int dimModel = nHeads * headDim;
        unsigned int hiddenStride = seqLen * dimModel;
        unsigned int threads = min(1024, round_up(dimModel, 32));
        dim3 gridDim(batchSize, seqLen, round_up(dimModel, threads) / threads);
        dim3 blockDim(threads, 1, 1);
        if (useHalf) {
            RoPEKernel<<<gridDim, blockDim, 0, 0>>>(
                dimModel,
                headDim,
                hiddenStride,
                seqLen,
                reinterpret_cast<const int64_t *>(posIDs),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<half *>(output));

        } else {
            RoPEKernel<<<gridDim, blockDim, 0, 0>>>(
                dimModel,
                headDim,
                hiddenStride,
                seqLen,
                reinterpret_cast<const int64_t *>(posIDs),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<float *>(output));
        }
    }
}// namespace refactor::kernel::cuda
