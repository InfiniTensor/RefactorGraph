#include "../../../src/kernels/transpose/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, TransposeCpu) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{1, 3, 5, 7});
    auto transposed = Tensor::share(DataType::F32, Shape{5, 7, 1, 3});
    auto kernel = TransposeCpu::build(TransposeInfo(dataTensor->shape, Permutation{2, 3, 0, 1}));
    ASSERT_TRUE(kernel);
    auto routine = kernel->lower();
    // put input data
    std::vector<float>
        data(dataTensor->elementsSize()),
        out(transposed->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
    // inference
    auto res = runtime::Resources();
    void const *inputs[]{data.data()};
    void *outputs[]{out.data()};
    routine(res, inputs, outputs);
    // check
    for (auto x : range0_(out.size())) {
        fmt::print("{} ", x);
    }
    fmt::println("");
}
