#ifndef KERNEL_CUDA_THREADS_DISTRIBUTER_CUH
#define KERNEL_CUDA_THREADS_DISTRIBUTER_CUH

namespace refactor::kernel::cuda {

    class ThreadsDistributer {
        int _maxGridSize;

    public:
        struct GridLayout {
            int gridSize, blockSize;
        };

        ThreadsDistributer();

        GridLayout operator()(unsigned long long n) const noexcept;
    };

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_THREADS_DISTRIBUTER_CUH
