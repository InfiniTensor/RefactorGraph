#include "../../../src/kernels/where/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, WhereCpu) {
    // build routine
    auto cTensor = Tensor::share(DataType::Bool, Shape{2, 5});
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto yTensor = Tensor::share(DataType::F32, Shape{3, 2, 5});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto kernel = WhereCpu::build({*cTensor, *xTensor, *yTensor});
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put inputs data
    bool dataC[cTensor->elementsSize()];
    std::generate_n(dataC, cTensor->elementsSize(), []() { return true; });
    std::vector<float>
        dataX(xTensor->elementsSize()),
        dataY(yTensor->elementsSize()),
        result(outTensor->elementsSize());
    for (auto i : range0_(dataX.size())) { dataX[i] = 7; }
    for (auto i : range0_(dataY.size())) { dataY[i] = 3; }
    // inference
    {
        void const *inputs[]{dataC, dataX.data(), dataY.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto x : result) {
        EXPECT_FLOAT_EQ(7, x);
    }
}
