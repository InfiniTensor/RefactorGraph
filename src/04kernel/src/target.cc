#include "kernel/target.h"
#include "refactor/common.h"
#include "utilities/cuda/cuda_mem.h"
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
                return cuda::memFunc();
#else
                UNREACHABLE();
#endif
            }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
