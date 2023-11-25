#include "../../../src/kernels/where/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, WhereCpu) {
    // build routine
    auto cTensor = Tensor::share(DataType::Bool, Shape{2, 5});
    auto xTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto yTensor = Tensor::share(DataType::F32, Shape{3, 2, 5});
    auto outTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 5});
    auto kernel = WhereCpu::build({*cTensor, *xTensor, *yTensor});
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // malloc
    auto mfn = Target(Target::Cpu).memManager();
    auto mc = hardware::ForeignBlob::share(mfn, cTensor->bytesSize());
    auto mx = hardware::ForeignBlob::share(mfn, xTensor->bytesSize());
    auto my = hardware::ForeignBlob::share(mfn, yTensor->bytesSize());
    auto mout = hardware::ForeignBlob::share(mfn, outTensor->bytesSize());
    // put inputs data
    int dataC[cTensor->elementsSize()];
    memset(dataC, 1, cTensor->elementsSize() * sizeof(bool));
    mc->copyIn(dataC, cTensor->bytesSize());
    std::vector<float> data(xTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 7; }
    mx->copyIn(data.data(), xTensor->bytesSize());
    data.resize(yTensor->elementsSize());
    for (auto i : range0_(data.size())) { data[i] = 3; }
    my->copyIn(data.data(), yTensor->bytesSize());
    // inference
    void const *inputs[]{*mc, *mx, *my};
    void *outputs[]{*mout};
    routine(res, nullptr, inputs, outputs);
    // take output data
    std::vector<float> result(outTensor->elementsSize());
    mout->copyOut(result.data(), outTensor->bytesSize());
    // check
    for (auto x : result) {
        EXPECT_FLOAT_EQ(7, x);
    }
}
