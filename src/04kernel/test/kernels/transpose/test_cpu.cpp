#include "../../../src/kernels/transpose/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, TransposeCpu) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{1, 3, 2, 5});
    auto kernel = TransposeCpu::build(
        TransposeInfo(
            dataTensor->dataType,
            dataTensor->shape,
            Permutation{2, 3, 0, 1}));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        data(dataTensor->elementsSize()),
        out(data.size());
    std::iota(data.begin(), data.end(), 0);
    // inference
    void const *inputs[]{data.data()};
    void *outputs[]{out.data()};
    routine(res, nullptr, inputs, outputs);
    // check
    fmt::println("{}", vec2str(data));
    fmt::println("{}", vec2str(out));
}
