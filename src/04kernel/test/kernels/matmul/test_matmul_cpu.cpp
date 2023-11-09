#include "../src/kernels/mat_mul/cpu_kernel.hh"
#include "kernel/target.h"
#include <gtest/gtest.h>
using namespace refactor;
using namespace kernel;

TEST(kernel, MatMulCPU_WithBias) {
    // build routine
    auto A = Tensor::share(DataType::F32, Shape{1, 2, 2});
    auto B = Tensor::share(DataType::F32, Shape{2, 2});
    auto C = Tensor::share(DataType::F32, Shape{});
    auto Y = Tensor::share(DataType::F32, Shape{2, 2});
    MatMulInfo info(*A, *B, *C);
    auto kernel = MatMulCPU::build(*A, *B, *Y, info);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto ma = mem_manager::ForeignBlob::share(mfn, A->bytesSize());
    auto mb = mem_manager::ForeignBlob::share(mfn, B->bytesSize());
    auto mc = mem_manager::ForeignBlob::share(mfn, C->bytesSize());
    auto my = mem_manager::ForeignBlob::share(mfn, Y->bytesSize());
    // put input data
    std::vector<float> dataA{1.0, 2.0, 0.0, 0.5};
    std::vector<float> dataB{1.0, 2.0, 0.0, 0.5};
    std::vector<float> dataC{1.0};
    std::vector<float> ans{2, 4, 1, 1.25};
    ma->copyIn(dataA.data(), A->bytesSize());
    mb->copyIn(dataB.data(), B->bytesSize());
    mc->copyIn(dataC.data(), C->bytesSize());
    // inference
    void const *inputs[]{*ma, *mb, *mc};
    void *outputs[]{*my};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(Y->elementsSize());
    my->copyOut(result.data(), Y->bytesSize());
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
    MatMulInfo info(*A, *B);
    auto kernel = MatMulCPU::build(*A, *B, *Y, info);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto ma = mem_manager::ForeignBlob::share(mfn, A->bytesSize());
    auto mb = mem_manager::ForeignBlob::share(mfn, B->bytesSize());
    auto my = mem_manager::ForeignBlob::share(mfn, Y->bytesSize());
    // put input data
    std::vector<uint16_t> dataA{3, 2, 0, 1};
    std::vector<uint16_t> dataB{1, 0, 2, 3};
    std::vector<uint16_t> ans{7, 6, 2, 3};
    ma->copyIn(dataA.data(), A->bytesSize());
    mb->copyIn(dataB.data(), B->bytesSize());

    // inference
    void const *inputs[]{*ma, *mb};
    void *outputs[]{*my};
    routine(res, inputs, outputs);
    // take output data
    std::vector<uint16_t> result(Y->elementsSize());
    my->copyOut(result.data(), Y->bytesSize());
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
    MatMulInfo info(*A, *B, *C);
    auto kernel = MatMulCPU::build(*A, *B, *Y, info);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto ma = mem_manager::ForeignBlob::share(mfn, A->bytesSize());
    auto mb = mem_manager::ForeignBlob::share(mfn, B->bytesSize());
    auto mc = mem_manager::ForeignBlob::share(mfn, C->bytesSize());
    auto my = mem_manager::ForeignBlob::share(mfn, Y->bytesSize());
    // put input data
    std::vector<float> dataA{1.0, 2.0, 0.0, 0.5,
                             1.0, 0.0, 0.0, 1.0};
    std::vector<float> dataB{1.0, 2.0, 0.0, 0.5,
                             1.0, 0.0, 0.0, 1.0};
    std::vector<float> dataC{1.0, 0.0};
    std::vector<float> ans{2.0, 4.0, 0.0, 0.25,
                           2.0, 3.0, 0.0, 0.5,
                           2.0, 3.0, 0.0, 0.5,
                           2.0, 1.0, 0.0, 1.0};
    ma->copyIn(dataA.data(), A->bytesSize());
    mb->copyIn(dataB.data(), B->bytesSize());
    mc->copyIn(dataC.data(), C->bytesSize());
    // inference
    void const *inputs[]{*ma, *mb, *mc};
    void *outputs[]{*my};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(Y->elementsSize());
    my->copyOut(result.data(), Y->bytesSize());
    // check
    for (auto i = 0; i < result.size(); i++) {
        EXPECT_FLOAT_EQ(result[i], ans[i]);
    }
}
