#include "../../../src/kernels/rms_normalization/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, RmsNormalizationCpu) {
    // build routine
    auto y = Tensor::share(DataType::F32, Shape{2, 3, 4});
    auto x = Tensor::share(DataType::F32, Shape{2, 3, 4});
    auto w = Tensor::share(DataType::F32, Shape{4});
    auto kernel = RmsNormalizationCpu::build(0, *x);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> y_(y->elementsSize());
    std::vector<float> x_(x->elementsSize());
    std::vector<float> w_(w->elementsSize());
    std::iota(x_.begin(), x_.end(), 0);
    std::iota(w_.begin(), w_.end(), 1);
    // inference
    {
        void const *inputs[]{x_.data(), w_.data()};
        void *outputs[]{y_.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(2 * 3)) {
        auto x__ = x_.data() + i * 4;
        auto acc = std::accumulate(x__, x__ + 4, 0.f, [&](auto acc, auto it) {
            return acc + it * it;
        });
        auto rms = 1. / std::sqrt(acc / 4);
        for (auto j : range0_(4)) {
            EXPECT_FLOAT_EQ(y_[i * 4 + j], x_[i * 4 + j] * rms * w_[j]);
        }
    }
}
