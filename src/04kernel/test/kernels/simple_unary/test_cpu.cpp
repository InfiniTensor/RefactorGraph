#include "../../../src/kernels/simple_unary/cpu_kernel.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

static void testOp(SimpleUnaryType opType, float check(float)) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = SimpleUnaryCpu::build(opType, *dataTensor);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // malloc
    auto cpuMem = mem_manager::ForeignBlob::share(
        Target(Target::Cpu).memManager(),
        dataTensor->bytesSize());
    // put input data
    std::vector<float> data(dataTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = i * 1e-4f; }
    cpuMem->copyIn(data.data(), dataTensor->bytesSize());
    // inference
    void const *inputs[]{*cpuMem};
    void *outputs[]{*cpuMem};
    routine(res, nullptr, inputs, outputs);
    // take output data
    std::vector<float> result(dataTensor->elementsSize());
    cpuMem->copyOut(result.data(), dataTensor->bytesSize());
    // check
    for (auto i : range0_(data.size())) {
        EXPECT_FLOAT_EQ(check(data[i]), result[i]);
    }
}

TEST(kernel, SimpleUnaryCpu) {
    testOp(SimpleUnaryType::Abs, std::abs);
    testOp(SimpleUnaryType::Sqrt, std::sqrt);
    testOp(SimpleUnaryType::Tanh, std::tanh);
}
