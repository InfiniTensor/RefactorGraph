#include "kernel/target.h"
#include "common/error_handler.h"
#include "cuda_mem.h"
#include <cstdlib>
#include <cstring>

namespace refactor::kernel {

    MemFunctions const &Target::memFunc() const {
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
            case NvidiaGpu: {
#ifdef USE_CUDA
                static MemFunctions const F{
                    cuda::malloc,
                    cuda::free,
                    cuda::memcpy_h2d,
                    cuda::memcpy_d2h,
                    cuda::memcpy_d2d,
                };
                return F;
#else
                UNREACHABLE();
#endif
            }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
