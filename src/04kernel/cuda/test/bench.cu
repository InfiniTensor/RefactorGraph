#include "common.h"
#include "kernel/cuda/bench.cuh"
#include <chrono>
#include <execution>
#include <gtest/gtest.h>

using namespace std::chrono;
using namespace refactor;
using namespace kernel;

namespace cpu_ {

    static void sigmoid(float *out, const float *in, size_t n) {
        std::transform(
            std::execution::par_unseq,
            in, in + n, out,
            [](auto const x) { return 1.0f / (1.0f + std::exp(-x)); });
    }

}// namespace cpu_

TEST(bench, unique) {
    constexpr static size_t N = 65536;
    static float src[N], cpu[N], gpu[N];

    auto t0 = high_resolution_clock::now();
    for (auto i : range0_(1000))
        cpu_::sigmoid(cpu, src, N);
    auto t1 = high_resolution_clock::now();
    fmt::println("cpu: {} ns", duration_cast<microseconds>(t1 - t0).count());

    cuda::sigmoid(gpu, src, N);

    for (auto i : range0_(N)) {
        EXPECT_FLOAT_EQ(cpu[i], gpu[i]);
    }
}
