#include "kernel/target.h"
#include "common.h"
#include "utilities/cuda/cuda_mem.cuh"
#include <cstdlib>
#include <cstring>

namespace refactor::kernel {

    Arc<MemManager> Target::memManager() const {
        switch (internal) {
            case Cpu: {
                class BasicCpuMemManager final : public mem_manager::MemManager {
                public:
                    static Arc<mem_manager::MemManager> instance() {
                        static auto I = std::make_shared<BasicCpuMemManager>();
                        return I;
                    }
                    void *malloc(size_t bytes) noexcept final {
                        return std::malloc(bytes);
                    }
                    void free(void *ptr) noexcept final {
                        std::free(ptr);
                    }
                    void *copyHD(void *dst, void const *src, size_t bytes) const noexcept final {
                        return std::memcpy(dst, src, bytes);
                    }
                    void *copyDH(void *dst, void const *src, size_t bytes) const noexcept final {
                        return std::memcpy(dst, src, bytes);
                    }
                    void *copyDD(void *dst, void const *src, size_t bytes) const noexcept final {
                        return std::memcpy(dst, src, bytes);
                    }
                };
                return BasicCpuMemManager::instance();
            }
#ifdef USE_CUDA
            case NvidiaGpu: {
                return cuda::BasicCudaMemManager::instance();
            }
#endif
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
