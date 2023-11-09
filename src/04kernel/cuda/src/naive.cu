#include "common.h"
#include "kernel/cuda/naive.cuh"
#include "kernel/cuda/threads_distributer.cuh"
#include <chrono>
#include <thrust/device_vector.h>

__global__ static void sigmoidKernel(float *out, const float *in, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0 / (1.0 + expf(-in[tid]));
    }
}

namespace refactor::kernel::cuda {

    void sigmoid(float *out, float const *in, unsigned long long n) {
        thrust::device_vector<float>
            in_(in, in + n),
            out_(n);

        ThreadsDistributer distributer;
        auto [grid, block] = distributer(n);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < 1000; ++i)
            sigmoidKernel<<<grid, block>>>(out_.data().get(), in_.data().get(), n);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fmt::println("gpu: {} ns", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        cudaMemcpy(out, out_.data().get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    }

}// namespace refactor::kernel::cuda
