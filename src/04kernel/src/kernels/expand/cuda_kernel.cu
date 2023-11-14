#include "cuda_kernel.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    Routine ExpandCuda::lower(Resources &) const noexcept {
        return [](Resources &res, void const **inputs, void **outputs) {
        };
    }

}// namespace refactor::kernel
