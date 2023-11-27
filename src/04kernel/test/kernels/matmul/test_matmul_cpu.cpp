#include "../src/kernels/mat_mul/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, MatMulCPU_WithBias) {
    // build routine
    auto A = Tensor::share(DataType::F32, Shape{1, 2, 2});
    auto B = Tensor::share(DataType::F32, Shape{2, 2});
    auto C = Tensor::share(DataType::F32, Shape{});
    auto Y = Tensor::share(DataType::F32, Shape{2, 2});
    auto kernel = MatMulCPU::build(MatMulInfo(*A, *B, *C, false, false, 1, 1));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        dataA{1.0, 2.0, 0.0, 0.5},
        dataB{1.0, 2.0, 0.0, 0.5},
        dataC{1.0},
        result(Y->elementsSize()),
        ans{2, 4, 1, 1.25};
    // inference
    {
        void const *inputs[]{dataA.data(), dataB.data(), dataC.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i = 0; i < result.size(); i++) {
        EXPECT_FLOAT_EQ(result[i], ans[i]);
    }
}

TEST(kernel, MatMulCPU_UINT16NoBias) {
    // build routine
    auto A = Tensor::share(DataType::U16, Shape{2, 2});
    auto B = Tensor::share(DataType::U16, Shape{2, 2});
    auto Y = Tensor::share(DataType::U16, Shape{2, 2});
    auto kernel = MatMulCPU::build(MatMulInfo(*A, *B, std::nullopt, false, false, 1, 1));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<uint16_t>
        dataA{3, 2, 0, 1},
        dataB{1, 0, 2, 3},
        result(Y->elementsSize()),
        ans{7, 6, 2, 3};
    // inference
    {
        void const *inputs[]{dataA.data(), dataB.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], ans[i]);
    }
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
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>
        dataA{1.0, 2.0, 0.0, 0.5,
              1.0, 0.0, 0.0, 1.0},
        dataB{1.0, 2.0, 0.0, 0.5,
              1.0, 0.0, 0.0, 1.0},
        dataC{1.0, 0.0},
        result(Y->elementsSize()),
        ans{2.0, 4.0, 0.0, 0.25,
            2.0, 3.0, 0.0, 0.5,
            2.0, 3.0, 0.0, 0.5,
            2.0, 1.0, 0.0, 1.0};
    // inference
    {
        void const *inputs[]{dataA.data(), dataB.data(), dataC.data()};
        void *outputs[]{result.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i = 0; i < result.size(); i++) {
        EXPECT_FLOAT_EQ(result[i], ans[i]);
    }
}
