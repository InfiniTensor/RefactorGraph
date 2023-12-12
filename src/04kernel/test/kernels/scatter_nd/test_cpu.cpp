#include "../../../src/kernels/scatter_nd/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, ScatterNDCpu) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{8});
    auto indices = Tensor::share(DataType::I64, Shape{4, 1});
    auto updates = Tensor::share(DataType::F32, Shape{4});
    auto kernel = ScatterNDCpu::build(ScatterNDInfo(*data, *indices));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> data_(data->elementsSize());
    std::iota(data_.begin(), data_.end(), 1);
    std::vector<int64_t> indices_{4, 3, 1, 7};
    std::vector<float> updates_{9, 10, 11, 12};
    // inference
    {
        void const *inputs[]{data_.data(), indices_.data(), updates_.data()};
        void *outputs[]{data_.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    EXPECT_EQ(data_, (decltype(data_){1, 11, 3, 10, 9, 6, 7, 12}));
}
