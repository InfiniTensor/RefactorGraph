#include "kernel/target.h"
#include "refactor/common.h"
#include "utilities/cuda/cuda_mem.cuh"
#include <cstdlib>
#include <cstring>

namespace refactor::kernel {

    MemFunctions Target::memFunc() const {
        switch (internal) {
            case Cpu: {
                static MemFunctions const F{
                    std::malloc,
                    std::free,
                    std::memcpy,
                    std::memcpy,
                    std::memcpy,
                };
                return F;
            }
#ifdef USE_CUDA
            case NvidiaGpu: {
                return {
                    cuda::malloc,
                    cuda::free,
                    cuda::memcpy_h2d,
                    cuda::memcpy_d2h,
                    cuda::memcpy_d2d,
                };
            }
#endif
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
