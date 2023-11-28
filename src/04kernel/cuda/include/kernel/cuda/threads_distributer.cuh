#ifndef KERNEL_CUDA_THREADS_DISTRIBUTER_CUH
#define KERNEL_CUDA_THREADS_DISTRIBUTER_CUH

namespace refactor::kernel::cuda {

    /// @brief 内核的启动参数。
    struct KernelLaunchParameters {
        /// @brief 网格中块的数量和块中线程的数量。
        int gridSize, blockSize;
        /// @brief 要处理任务总量。
        size_t n;
        /// @brief 用于执行内核的流。
        void *stream;
    };

    class ThreadsDistributer {
        int _maxGridSize;

    public:
        ThreadsDistributer();

        KernelLaunchParameters operator()(size_t n) const;
    };

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_THREADS_DISTRIBUTER_CUH
