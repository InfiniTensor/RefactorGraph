#include "../../../include/kernel/attributes/pad_info.h"
#include "../../../src/kernels/pad/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, PadCpu) {
    // no constant_value
    {
        PadDimension dims{
            {2, 4, 1},
            {3, 5, 1},
        };
        // build routine
        auto xTensor = Tensor::share(DataType::F32, Shape{2, 3});
        auto yTensor = Tensor::share(DataType::F32, Shape{4, 5});
        PadType mode = PadType::Constant;
        auto kernel = PadCpu::build(PadInfo(dims, *xTensor), mode, std::nullopt);
        ASSERT_TRUE(kernel);
        auto res = runtime::Resources();
        auto routine = kernel->lower(res).routine;
        // set input data
        std::vector<float>
            data(xTensor->elementsSize(), 1),
            result(yTensor->elementsSize());
        // inference
        {
            void const *inputs[]{data.data()};
            void *outputs[]{result.data()};
            routine(res, nullptr, inputs, outputs);
        }
        // check
        std::vector<float> output = {0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.};
        for (auto i : range0_(result.size())) {
            EXPECT_FLOAT_EQ(output[i], result[i]);
        }
    }
    // have constant_value
    {
        PadDimension dims{
            {2, 4, 1},
            {3, 5, 1},
        };
        // build routine
        auto t1Tensor = Tensor::share(DataType::F32, Shape{2, 3});
        auto t2Tensor = Tensor::share(DataType::I64, Shape{4});
        auto t3Tensor = Tensor::share(DataType::F32, Shape{});
        auto yTensor = Tensor::share(DataType::F32, Shape{4, 5});
        PadType type = PadType::Constant;
        auto kernel = PadCpu::build(PadInfo(dims, *t1Tensor), type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        ASSERT_TRUE(kernel);
        auto res = runtime::Resources();
        auto routine = kernel->lower(res).routine;
        // set input data
        std::vector<float>
            data(t1Tensor->elementsSize(), 1),
            result(yTensor->elementsSize());
        std::vector<float> constant_value(1, 1.2);
        std::vector<int64_t> pads_value(4, 1);
        // inference
        {
            void const *inputs[]{data.data(), pads_value.data(), constant_value.data()};
            void *outputs[]{result.data()};
            routine(res, nullptr, inputs, outputs);
        }
        // check
        std::vector<float> output = {1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1., 1., 1., 1.2, 1.2, 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.2, 1.2};
        for (auto i : range0_(result.size())) {
            EXPECT_FLOAT_EQ(output[i], result[i]);
        }
    }
    {
        PadDimension dims{
            {2, 4, 1},
            {3, 5, 1},
            {1, 1, 0},
            {4, 8, 2},
        };
        // build routine
        auto t1Tensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 4});
        auto t2Tensor = Tensor::share(DataType::I64, Shape{8});
        auto t3Tensor = Tensor::share(DataType::F32, Shape{});
        auto yTensor = Tensor::share(DataType::F32, Shape{4, 5, 1, 8});
        PadType type = PadType::Constant;
        auto kernel = PadCpu::build(PadInfo(dims, *t1Tensor), type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        ASSERT_TRUE(kernel);
        auto res = runtime::Resources();
        auto routine = kernel->lower(res).routine;
        // set input data
        std::vector<float>
            data(t1Tensor->elementsSize(), 1),
            result(yTensor->elementsSize());
        std::vector<float> constant_value(1, 1.2);
        std::vector<int64_t> pads_value{1, 1, 0, 2, 1, 1, 0, 2};
        // inference
        {
            void const *inputs[]{data.data(), pads_value.data(), constant_value.data()};
            void *outputs[]{result.data()};
            routine(res, nullptr, inputs, outputs);
        }
        // check
        std::vector<float> output = {1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.0000, 1.0000, 1.0000, 1.0000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.0000, 1.0000, 1.0000, 1.0000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.0000, 1.0000, 1.0000, 1.0000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.0000, 1.0000, 1.0000, 1.0000, 1.2000, 1.2000, 1.2000, 1.2000, 1.0000,
                                     1.0000, 1.0000, 1.0000, 1.2000, 1.2000, 1.2000, 1.2000, 1.0000, 1.0000,
                                     1.0000, 1.0000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000,
                                     1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000, 1.2000};
        for (auto i : range0_(result.size())) {
            EXPECT_FLOAT_EQ(output[i], result[i]);
        }
    }
}
