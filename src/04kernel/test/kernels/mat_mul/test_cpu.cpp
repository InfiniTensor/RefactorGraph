#include "../src/kernels/mat_mul/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

template<class T>
static void check(
    Resources &&res,
    Routine &&routine,
    std::vector<T> ans,
    std::vector<T> const &a,
    std::vector<T> const &b,
    std::vector<T> const &c) {
    std::vector<T> result(ans.size());
    // inference
    void const *inputs[]{a.data(), b.data(), c.data()};
    void *outputs[]{result.data()};
    routine(res, nullptr, inputs, outputs);
    // check
    EXPECT_EQ(result, ans);
}

TEST(kernel, MatMulCPU_WithBias) {
    // build routine
    auto A = Tensor::share(DataType::F32, Shape{1, 2, 2});
    auto B = Tensor::share(DataType::F32, Shape{2, 2});
    auto C = Tensor::share(DataType::F32, Shape{});
    auto Y = Tensor::share(DataType::F32, Shape{2, 2});
    auto kernel = MatMulCPU::build(MatMulInfo(*A, *B, *C, false, false, 1, 1));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    check<float>(std::move(res), kernel->lower(res).routine,
                 {2, 4, 1, 1.25},
                 {1.0, 2.0, 0.0, 0.5},
                 {1.0, 2.0, 0.0, 0.5},
                 {1.0});
}

TEST(kernel, MatMulCPU_UINT16NoBias) {
    // build routine
    auto A = Tensor::share(DataType::U16, Shape{2, 2});
    auto B = Tensor::share(DataType::U16, Shape{2, 2});
    auto Y = Tensor::share(DataType::U16, Shape{2, 2});
    auto kernel = MatMulCPU::build(MatMulInfo(*A, *B, std::nullopt, false, false, 1, 1));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    check<uint16_t>(std::move(res), kernel->lower(res).routine,
                    {7, 6, 2, 3},
                    {3, 2, 0, 1},
                    {1, 0, 2, 3},
                    {});
}

TEST(kernel, MatMulCPU_Broadcast) {
    // build routine
    auto A = Tensor::share(DataType::F32, Shape{2, 1, 2, 2});
    auto B = Tensor::share(DataType::F32, Shape{1, 2, 2, 2});
    auto C = Tensor::share(DataType::F32, Shape{2, 1});
    auto Y = Tensor::share(DataType::F32, Shape{2, 2, 2, 2});
    auto kernel = MatMulCPU::build(MatMulInfo(*A, *B, *C, false, false, 1, 1));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    check<float>(std::move(res), kernel->lower(res).routine,
                 {2.0, 4.0, 0.0, 0.25,
                  2.0, 3.0, 0.0, 0.5,
                  2.0, 3.0, 0.0, 0.5,
                  2.0, 1.0, 0.0, 1.0},
                 {1.0, 2.0, 0.0, 0.5,
                  1.0, 0.0, 0.0, 1.0},
                 {1.0, 2.0, 0.0, 0.5,
                  1.0, 0.0, 0.0, 1.0},
                 {1.0, 0.0});
}
