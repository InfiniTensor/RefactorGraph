#include "../../../src/kernels/concat/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, ConcatCpu) {
    // build routine
    std::vector<Arc<Tensor>> inputTensors{
        Tensor::share(DataType::F32, Shape{2, 3, 1, 1, 7, 7}),// 勿
        Tensor::share(DataType::F32, Shape{2, 3, 1, 9, 7, 7}),// 忘
        Tensor::share(DataType::F32, Shape{2, 3, 1, 3, 7, 7}),// 国
        Tensor::share(DataType::F32, Shape{2, 3, 1, 7, 7, 7}),// 耻
    };
    auto result = Tensor::share(DataType::F32, Shape{2, 3, 1, 20, 7, 7});
    TensorRefs inputs_;
    inputs_.reserve(inputTensors.size());
    std::transform(inputTensors.begin(), inputTensors.end(),
                   std::back_inserter(inputs_),
                   [](auto const &it) { return std::cref(*it); });
    auto kernel = ConcatCpu::build(SplitInfo(3, inputs_));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res);
    // put input data
    std::vector<float>
        ins[]{
            std::vector<float>(inputTensors[0]->elementsSize()),
            std::vector<float>(inputTensors[1]->elementsSize()),
            std::vector<float>(inputTensors[2]->elementsSize()),
            std::vector<float>(inputTensors[3]->elementsSize()),
        },
        out(result->elementsSize());
    std::iota(ins[0].begin(), ins[0].end(), 0);
    std::iota(ins[1].begin(), ins[1].end(), 0);
    std::iota(ins[2].begin(), ins[2].end(), 0);
    std::iota(ins[3].begin(), ins[3].end(), 0);
    // inference
    void const *inputs[]{ins[0].data(), ins[1].data(), ins[2].data(), ins[3].data()};
    void *outputs[]{out.data()};
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
                EXPECT_EQ(out[i * offsets[4] + offsets[j] + k],
                          i * (offsets[j + 1] - offsets[j]) + k);
            }
        }
    }
}
