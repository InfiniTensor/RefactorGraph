#include "../../../src/kernels/softmax/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SoftmaxCpu) {
    // build routine
    auto xTensor = Tensor::share(DataType::F64, Shape{2, 3, 2, 4});
    auto yTensor = Tensor::share(DataType::F64, Shape{2, 3, 2, 4});
    dim_t axis = 1;
    auto kernel = SoftmaxCpu::build(SoftmaxInfo(*xTensor, axis));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // set input data
    std::vector<double>
        data(xTensor->elementsSize(), 0),
        result(yTensor->elementsSize());
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto x : result) {
        EXPECT_DOUBLE_EQ(1.0 / 3, x);
    }
}
