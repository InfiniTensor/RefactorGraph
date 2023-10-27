#include "../../../src/kernels/split/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, SplitCpu) {
    // build routine
    auto dataTensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 20, 7, 7});
    std::vector<Arc<Tensor>> outputTensors{
        Tensor::share(DataType::F32, Shape{2, 3, 1, 1, 7, 7}),// 勿
        Tensor::share(DataType::F32, Shape{2, 3, 1, 9, 7, 7}),// 忘
        Tensor::share(DataType::F32, Shape{2, 3, 1, 3, 7, 7}),// 国
        Tensor::share(DataType::F32, Shape{2, 3, 1, 7, 7, 7}),// 耻
    };
    TensorRefs outputs_;
    outputs_.reserve(outputTensors.size());
    std::transform(outputTensors.begin(), outputTensors.end(),
                   std::back_inserter(outputs_),
                   [](auto const &it) { return std::cref(*it); });
    auto kernel = SplitCpu::build(SplitInfo(3, outputs_));
    ASSERT_TRUE(kernel);
    auto routine = kernel->lower();
    // put input data
    std::vector<float>
        data(dataTensor->elementsSize()),
        outs[]{
            std::vector<float>(outputTensors[0]->elementsSize()),
            std::vector<float>(outputTensors[1]->elementsSize()),
            std::vector<float>(outputTensors[2]->elementsSize()),
            std::vector<float>(outputTensors[3]->elementsSize()),
        };
    std::iota(data.begin(), data.end(), 0);
    // inference
    auto res = runtime::Resources();
    void const *inputs[]{data.data()};
    void *outputs[]{outs[0].data(), outs[1].data(), outs[2].data(), outs[3].data()};
    routine(res, inputs, outputs);
    // check
    constexpr static size_t
        loops[]{
            1 * 7 * 7,
            9 * 7 * 7,
            3 * 7 * 7,
            7 * 7 * 7,
        },
        offsets[]{
            0 * 7 * 7,
            1 * 7 * 7,
            10 * 7 * 7,
            13 * 7 * 7,
            20 * 7 * 7,
        };
    for (auto i : range0_(2 * 3)) {
        for (auto j : range0_(4)) {
            for (auto k : range0_(loops[j])) {
                EXPECT_EQ(outs[j][i * loops[j] + k], (i * offsets[4] + offsets[j]) + k);
            }
        }
    }
}
