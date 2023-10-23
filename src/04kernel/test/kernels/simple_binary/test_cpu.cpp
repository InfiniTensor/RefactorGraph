#include "../../../src/kernels/simple_binary/basic_cpu.hh"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, BinaryBasicCpu) {
    // build routine
    auto a = Tensor::share(DataType::F32, Shape{20, 30, 1});
    auto b = Tensor::share(DataType::F32, Shape{30, 50});
    auto c = Tensor::share(DataType::F32, Shape{20, 30, 50});
    auto kernel = BinaryBasicCpu::build(SimpleBinaryType::Add, *a, *b);
    ASSERT_TRUE(kernel);
    auto routine = kernel->lower();
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto ma = mem_manager::ForeignBlob::share(mfn, a->bytesSize());
    auto mb = mem_manager::ForeignBlob::share(mfn, b->bytesSize());
    auto mc = mem_manager::ForeignBlob::share(mfn, c->bytesSize());
    // put input data
    std::vector<float> data(a->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 11; }
    ma->copyIn(data.data(), a->bytesSize());
    data.resize(b->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 7; }
    mb->copyIn(data.data(), b->bytesSize());
    // inference
    auto res = runtime::Resources();
    void const *inputs[]{*ma, *mb};
    void *outputs[]{*mc};
    routine(res, inputs, outputs);
    // take output data
    std::vector<float> result(c->elementsSize());
    mc->copyOut(result.data(), c->bytesSize());
    // check
    for (auto x : result) {
        EXPECT_FLOAT_EQ(18, x);
    }
}
