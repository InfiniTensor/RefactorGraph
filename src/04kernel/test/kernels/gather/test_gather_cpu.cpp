#include "../src/kernels/gather/cpu_kernel.hh"
#include <gtest/gtest.h>
using namespace refactor;
using namespace kernel;

TEST(kernel, GatherCPU) {
    // Case axis = 0, indexType= int64
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 2}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I64, Shape{2, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{2, 2, 2}, LayoutType::NCHW);
        auto cpuKernel = GatherCpu::build(GatherInfo(0, *data, *indices));
        ASSERT_TRUE(cpuKernel);
        auto res = runtime::Resources();
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 2.3, 3.4, 4.5, 5.7};
        std::vector<int64_t> b{0, 1, 1, 2};
        std::vector<float> c(output->elementsSize());
        // Compute
        {
            void const *inputs[]{a.data(), b.data()};
            void *outputs[]{c.data()};
            cpuRoutine(res, nullptr, inputs, outputs);
        }
        // Compare
        std::vector<float> ans{1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7};
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], ans[i]);
        }
    }

    // Case axis = 1, indexType= int32
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 3}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I32, Shape{1, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{3, 1, 2}, LayoutType::NCHW);
        auto cpuKernel = GatherCpu::build(GatherInfo(1, *data, *indices));
        ASSERT_TRUE(cpuKernel);
        auto res = runtime::Resources();
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9};
        std::vector<int32_t> b{0, 2};
        std::vector<float> c(output->elementsSize());
        // Compute
        {
            void const *inputs[]{a.data(), b.data()};
            void *outputs[]{c.data()};
            cpuRoutine(res, nullptr, inputs, outputs);
        }
        // Compare
        std::vector<float> ans{1.0, 1.9, 2.3, 3.9, 4.5, 5.9};
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], ans[i]);
        }
    }
}
