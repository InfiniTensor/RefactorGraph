#include "../src/kernels/mat_mul_integer/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, MatMulIntegerCpu) {
    // build routine
    auto A = Tensor::share(DataType::U8, Shape{2, 3});
    auto B = Tensor::share(DataType::U8, Shape{3, 1});
    auto Y = Tensor::share(DataType::I32, Shape{2, 1});
    auto kernel = MatMulIntegerCpu::build(MatMulIntegerInfo(TensorRefs{*A, *B}));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    // put input data
    std::vector<uint8_t>
        dataA{1, 2, 3, 4, 5, 6},
        dataB{1, 2, 3},
        workspace(workspaceSize);
    std::vector<int32_t>
        result(Y->elementsSize()),
        ans{14, 32};
    // inference
    {
        void const *inputs[]{dataA.data(), dataB.data()};
        void *outputs[]{result.data()};
        routine(res, workspace.data(), inputs, outputs);
    }
    // check
    EXPECT_EQ(result, ans);
}
