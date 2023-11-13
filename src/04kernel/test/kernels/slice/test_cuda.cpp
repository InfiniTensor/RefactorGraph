#ifdef USE_CUDA

#include "../../../src/kernels/slice/cpu_kernel.hh"
#include "../../../src/kernels/slice/cuda_kernel.hh"
#include "kernel/target.h"
#include "runtime/mem_manager.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SliceCuda) {
}

#endif
