#ifdef USE_CUDA

#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

#include "common.h"
#include "mem_manager/mem_manager.hh"

namespace refactor::kernel::cuda {

    class BasicCudaMemManager final : public mem_manager::MemManager {
    public:
        static Arc<mem_manager::MemManager> instance();
        void *malloc(size_t bytes) noexcept final;
        void free(void *ptr) noexcept final;
        void *copyHD(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDH(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDD(void *dst, void const *src, size_t bytes) const noexcept final;
    };

}// namespace refactor::kernel::cuda

#endif// CUDA_MEM_CUH

#endif// USE_CUDA
