#include "cuda_kernel.hh"
#include <cub/cub.cuh>

namespace refactor::kernel {
    using K = DynamicQuantizeLinearCuda;

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        using TI = float;
        using TO = uint8_t;

        return [size = size](Resources &, void *, void const *const *inputs, void *const *outputs) {
        };
    }

}// namespace refactor::kernel
