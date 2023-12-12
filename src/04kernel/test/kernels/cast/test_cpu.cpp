#include "../../../src/kernels/cast/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, CastCpu) {
    // build routine
    auto x = Tensor::share(DataType::F64, Shape{2, 3, 4, 5});
    auto y = Tensor::share(DataType::I8, Shape{2, 3, 4, 5});
    auto kernel = CastCpu::build(*x, *y);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<double> x_(x->elementsSize());
    std::vector<int8_t> y_(y->elementsSize());
    std::iota(x_.begin(), x_.end(), 0);
    // inference
    {
        void const *inputs[]{x_.data()};
        void *outputs[]{y_.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(static_cast<int8_t>(y->elementsSize()))) {
        EXPECT_EQ(y_[i], i);
    }
}
