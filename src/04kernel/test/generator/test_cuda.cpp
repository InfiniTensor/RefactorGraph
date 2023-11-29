#ifdef USE_CUDA

#include "../../src/generator/cuda_code_repo.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

constexpr static const char *code = R"~(
#include <cstdio>

__global__ void kernel() {
    printf("Hello World from GPU!\n");
}

extern "C" {
void launchKernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
}
)~";

TEST(generator, CudaCodeRepo) {
    CudaCodeRepo repo;
    auto function = repo.compile("helloWorld", code, "launchKernel");
    reinterpret_cast<void (*)()>(function)();
}

#endif// USE_CUDA
