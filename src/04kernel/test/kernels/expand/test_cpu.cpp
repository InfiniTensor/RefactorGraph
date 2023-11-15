#include "../../../src/kernels/expand/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, ExpandCpu) {
    // build routine
    auto input = Tensor::share(DataType::F32, {3, 4, 1, 6}),
         output = Tensor::share(DataType::F32, {2, 3, 4, 5, 6});
    auto kernel = ExpandCpu::build(ExpandInfo(*input, *output));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        result(output->elementsSize());
    std::iota(data.begin(), data.end(), 0);
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{result.data()};
        routine(res, inputs, outputs);
    }
    // check
    {
        auto idx = 0;
        for (auto i : range0_(2)) {
            for (auto j : range0_(12)) {
                for (auto k : range0_(5)) {
                    for (auto m : range0_(6)) {
                        ASSERT_EQ(result[idx++], j * 6 + m);
                    }
                }
            }
        }
    }
    // test reform
    auto kernelReformed = ExpandCpu::build(ExpandInfo(*input, *output).reform(16));
    ASSERT_TRUE(kernelReformed);
    auto routineReformed = kernelReformed->lower(res);
    std::vector<float> resultReformed(result.size());
    {
        void const *inputs[]{data.data()};
        void *outputs[]{resultReformed.data()};
        routineReformed(res, inputs, outputs);
    }
    // check
    ASSERT_EQ(result, resultReformed);
}
