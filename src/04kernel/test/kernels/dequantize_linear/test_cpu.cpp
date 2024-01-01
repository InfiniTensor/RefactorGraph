#include "../../../src/kernels/dequantize_linear/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, DequantizeLinearCpu) {
    // build routine
    auto x = Tensor::share(DataType::U8, {4});
    auto scale = Tensor::share(DataType::F32, {});
    auto zeroPoint = Tensor::share(DataType::U8, {});
    auto y = Tensor::share(DataType::F32, {4});
    auto kernel = DequantizeLinearCpu::build({*x, *scale, *zeroPoint}, *y);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<uint8_t> xData{0, 3, 128, 255};
    float scale_ = 2;
    uint8_t zp_ = 128;
    std::vector<float> yData(xData.size());
    // inference
    {
        void const *inputs[]{xData.data(), &scale_, &zp_};
        void *outputs[]{yData.data()};
        routine(res, nullptr, inputs, outputs);
    }
    // check
    ASSERT_EQ(yData, (decltype(yData){-256, -250, 0, 254}));
}
