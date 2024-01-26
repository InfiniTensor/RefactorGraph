#include "../../../src/kernels/slice/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SliceCpu_with1) {
    // build routine
    Dimensions dims{
        {0, 1, 1},
        {0, 1, 2},
        {0, 1, 1},
        {2, 1, 2},
    };
    auto input = Tensor::share(DataType::F32, Shape{1, 2, 1, 4}),
         output = Tensor::share(DataType::F32, Shape{1, 2, 1, 2});
    auto kernel = SliceCpu::build(SliceInfo(dims, *input));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        result(output->elementsSize());
    std::iota(data.begin(), data.end(), 0);
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> ans{2, 3, 6, 7};
    EXPECT_EQ(result, ans);
    // test reform
    auto kernelReformed = SliceCpu::build(SliceInfo(dims, *input).reform(16));
    ASSERT_TRUE(kernelReformed);
    auto routineReformed = kernelReformed->lower(res).routine;
    std::vector<float> resultReformed(result.size());
    {
        void const *inputs[]{data.data()};
        void *outputs[]{resultReformed.data()};
        routineReformed(res, nullptr, inputs, outputs);
    }
    // check
    EXPECT_EQ(resultReformed, ans);
}

TEST(kernel, SliceCpu) {
    // build routine
    Dimensions dims{
        {5, -2, 3},// 7 -> {5, 3, 1} -> {108, 900, -360}
        {2, 3, 2}, // 6 -> {2, 5}    -> { 36,  60,   90}
        {1, 1, 3}, // 5 -> {1, 2, 3} -> { 18,   6,   30}
        {0, 1, 1}, // 1 -> {0}
        {0, 1, 2}, // 2 -> {0, 1}
        {0, 1, 3}, // 3 -> {0, 1, 2}
    };
    auto input = Tensor::share(DataType::F32, Shape{7, 6, 5, 1, 2, 3}),
         output = Tensor::share(DataType::F32, Shape{3, 2, 3, 1, 2, 3});
    auto info = SliceInfo(dims, *input);
    auto kernel = SliceCpu::build(info);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        data(input->elementsSize()),
        result(output->elementsSize());
    std::iota(data.begin(), data.end(), 0);
    // inference
    {
        void const *inputs[]{data.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    dim_t
        di[]{5, 3, 1},
        dj[]{2, 5},
        dk[]{1, 2, 3};
    auto n = 6;
    for (auto i : range0_(3)) {
        for (auto j : range0_(2)) {
            for (auto k : range0_(3)) {
                // clang-format off
                auto src = di[i] * 6 * 5 * n + dj[j] * 5 * n + dk[k] * n;
                auto dst =    i  * 2 * 3 * n +    j  * 3 * n +    k  * n;
                // clang-format on
                for (auto l : range0_(n)) {
                    EXPECT_EQ(data[src + l], result[dst + l]);
                }
            }
        }
    }
    // test reform
    auto kernelReformed = SliceCpu::build(info.reform(16));
    ASSERT_TRUE(kernelReformed);
    auto routineReformed = kernelReformed->lower(res).routine;
    std::vector<float> resultReformed(result.size());
    {
        void const *inputs[]{data.data()};
        void *outputs[]{resultReformed.data()};
        routineReformed(res, nullptr, inputs, outputs);
    }
    // check
    ASSERT_EQ(result, resultReformed);
}
