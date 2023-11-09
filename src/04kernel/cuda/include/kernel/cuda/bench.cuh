#ifndef KERNEL_CUDA_BENCH_CUH
#define KERNEL_CUDA_BENCH_CUH

namespace refactor::kernel::cuda {

    void sigmoid(float *out, float const *in, unsigned long long n);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_BENCH_CUH
