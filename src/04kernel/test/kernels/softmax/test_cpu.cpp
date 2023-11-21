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
    auto routine = kernel->lower(res);
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto mx = mem_manager::ForeignBlob::share(mfn, xTensor->bytesSize());
    auto my = mem_manager::ForeignBlob::share(mfn, yTensor->bytesSize());
    std::vector<double> data(xTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 0; }
    mx->copyIn(data.data(), xTensor->bytesSize());
    // inference
    void const *inputs[]{*mx};
    void *outputs[]{*my};
    routine(res, nullptr, inputs, outputs);
    // take output data
    std::vector<double> result(yTensor->elementsSize());
    my->copyOut(result.data(), yTensor->bytesSize());
    // check
    for (auto x : result) {
        EXPECT_DOUBLE_EQ(1.0 / 3, x);
    }
}
