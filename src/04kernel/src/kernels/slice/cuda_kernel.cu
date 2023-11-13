#include "cuda_kernel.hh"
#include "kernel/cuda/split.cuh"
#include "mem_manager/foreign_blob.hh"
#include "runtime/mem_manager.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    Routine SliceCuda::lower(Resources &) const noexcept {
        return [](Resources &, void const **inputs, void **outputs) {
        };
    }

}// namespace refactor::kernel
