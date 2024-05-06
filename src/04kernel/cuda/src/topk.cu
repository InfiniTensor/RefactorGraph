#include "kernel/cuda/topk.cuh"
#include "macro.cuh"
#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace refactor::kernel::cuda {

using PairType = thrust::pair<float, uint32_t>;

struct ComparePair {
    __host__ __device__
    bool operator()(const PairType& a, const PairType& b) const {
        return a.first > b.first;
    }
};

/*
     __device__ 
    void process_element(unsigned int n, float  *__restrict__ dstVal,
        uint32_t  *__restrict__ dstIdx,
        PairType *list,
        uint32_t stride_axis,
        uint32_t init_offset){
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                step = blockDim.x * gridDim.x;
            tid < n;
            tid += step) {  
            uint32_t offset = init_offset + stride_axis * tid;
            dstVal[offset] = list[tid].first;
            dstIdx[offset] = list[tid].second;
         }
    }
*/



    __global__ static void TopKKernel(
        unsigned long long n,
        float const *__restrict__ data,
        float  *__restrict__ dstVal,
        uint32_t  *__restrict__ dstIdx,
        uint32_t topk,
        uint32_t stride_axis,
        uint32_t stride_in_pre,
        uint32_t stride_out_pre,
        unsigned int size) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {  
            PairType *list = new PairType[size];
            
            for(uint32_t i = 0; i < size; i++){                    
                uint32_t srcIdx = tid /stride_axis * stride_in_pre + tid % stride_axis + i * stride_axis;                    
                
                list[i] = PairType(data[srcIdx], i);
            }
            // thrust没有partial_sort算法，可尝试优化：分成size/topk组，每组取一个最大值
            thrust::sort(thrust::device, list, list + size, ComparePair());
            
            
            uint32_t init_offset = tid /stride_axis * stride_out_pre + tid % stride_axis;
            for (uint32_t i = 0; i < topk; i++)
            {
                uint32_t offset = init_offset + stride_axis * i;
                dstVal[offset] = list[i].first;
                dstIdx[offset] = list[i].second;
            }
            
            delete[] list;
        }
    }



    void launchTopK(
        KernelLaunchParameters const &params,
        float const *data,  float *dstVal, uint32_t *dstIdx,       
        uint32_t topk,
        uint32_t stride_axis,
        uint32_t stride_in_pre,
        uint32_t stride_out_pre,
        unsigned int size_axis) {    

            TopKKernel<<<
                params.gridSize,
                params.blockSize,
                0,
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                (data),
                (dstVal),
                (dstIdx),
                topk,
                stride_axis,
                stride_in_pre,
                stride_out_pre,
                size_axis);
        
    }

}// namespace refactor::kernel::cuda
