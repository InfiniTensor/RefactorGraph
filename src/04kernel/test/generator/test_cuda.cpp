#ifdef USE_CUDA

#include "../../src/generator/nvrtc_repo.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

constexpr static const char *code = R"~(
extern "C" __global__ void kernel() {
    printf("Hello World from GPU!\n");
}
)~";

TEST(generator, nvrtc) {
    auto handler = nvrtc::Handler::compile("helloWorld.cu", code, "kernel");
    CUDA_ASSERT(cuLaunchKernel(handler->kernel(),
                               1, 1, 1,
                               1, 1, 1,
                               0, nullptr, nullptr, nullptr));
}

#endif// USE_CUDA
