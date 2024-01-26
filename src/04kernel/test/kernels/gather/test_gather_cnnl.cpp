#ifdef USE_BANG

#include "../src/kernels/gather/cnnl_kernel.hh"
#include "../src/kernels/gather/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, GatherCnnl) {
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    // Case axis = 0, indexType= int32
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 2}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I32, Shape{2, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{2, 2, 2}, LayoutType::NCHW);
        GatherInfo info(0, *data, *indices);
        auto cnnlKernel = GatherCnnl::build(0, *data, *indices, *output);
        auto cpuKernel = GatherCpu::build(info);
        ASSERT_TRUE(cnnlKernel && cpuKernel);
        auto res = runtime::Resources();
        auto [cnnlRoutine, workspaceSize] = cnnlKernel->lower(res);
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 2.3, 3.4, 4.5, 5.7};
        std::vector<int> b{0, 1, 1, 2};
        std::vector<float> c(output->elementsSize());
        auto workspace = dev.malloc(workspaceSize),
             aMLU = dev.malloc(data->bytesSize()),
             bMLU = dev.malloc(indices->bytesSize()),
             cMLU = dev.malloc(output->bytesSize());
        aMLU->copyFromHost(a.data(), data->bytesSize());
        bMLU->copyFromHost(b.data(), indices->bytesSize());
        // Compute
        {
            void const *inputs[]{*aMLU, *bMLU};
            void *outputs[]{*cMLU};
            cnnlRoutine(res, *workspace, inputs, outputs);
            kernel::bang::sync();
        }
        {
            void const *inputs[]{a.data(), b.data()};
            void *outputs[]{c.data()};
            cpuRoutine(res, nullptr, inputs, outputs);
        }
        // Compare
        std::vector<float> result(output->elementsSize());
        cMLU->copyToHost(result.data(), output->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }

    // Case axis = 1, indexType= int32
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{3, 3}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I32, Shape{1, 2}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{3, 1, 2}, LayoutType::NCHW);
        GatherInfo info(1, *data, *indices);
        auto cnnlKernel = GatherCnnl::build(1, *data, *indices, *output);
        auto cpuKernel = GatherCpu::build(info);
        ASSERT_TRUE(cnnlKernel && cpuKernel);
        auto res = runtime::Resources();
        auto [cnnlRoutine, workspaceSize] = cnnlKernel->lower(res);
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a{1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9};
        std::vector<int> b{0, 2};
        std::vector<float> c(output->elementsSize());
        auto workspace = dev.malloc(workspaceSize),
             aMLU = dev.malloc(data->bytesSize()),
             bMLU = dev.malloc(indices->bytesSize()),
             cMLU = dev.malloc(output->bytesSize());
        aMLU->copyFromHost(a.data(), data->bytesSize());
        bMLU->copyFromHost(b.data(), indices->bytesSize());
        // Compute
        {
            void const *inputs[]{*aMLU, *bMLU};
            void *outputs[]{*cMLU};
            cnnlRoutine(res, *workspace, inputs, outputs);
            kernel::bang::sync();
        }
        {
            void const *inputs[]{a.data(), b.data()};
            void *outputs[]{c.data()};
            cpuRoutine(res, nullptr, inputs, outputs);
        }
        // Compare
        std::vector<float> result(output->elementsSize());
        cMLU->copyToHost(result.data(), output->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }

    // Case axis = 1, indexType= int32
    {
        // Create Tensor and build kernels
        auto data = Tensor::share(DataType::F32, Shape{32, 16}, LayoutType::NCHW);
        auto indices = Tensor::share(DataType::I64, Shape{1, 4}, LayoutType::NCHW);
        auto output = Tensor::share(DataType::F32, Shape{1, 4, 16}, LayoutType::NCHW);
        GatherInfo info(0, *data, *indices);
        auto cnnlKernel = GatherCnnl::build(0, *data, *indices, *output);
        auto cpuKernel = GatherCpu::build(info);
        ASSERT_TRUE(cnnlKernel && cpuKernel);
        auto res = runtime::Resources();
        auto [cnnlRoutine, workspaceSize] = cnnlKernel->lower(res);
        auto cpuRoutine = cpuKernel->lower(res).routine;
        // Init inputs and outputs
        std::vector<float> a;
        for (size_t i = 0; i < data->elementsSize(); i++) {
            a.push_back(i + 0.1f);
        }
        std::vector<int64_t> b(indices->elementsSize(), 0);
        std::vector<float> c(output->elementsSize());
        auto workspace = dev.malloc(workspaceSize),
             aMLU = dev.malloc(data->bytesSize()),
             bMLU = dev.malloc(indices->bytesSize()),
             cMLU = dev.malloc(output->bytesSize());
        aMLU->copyFromHost(a.data(), data->bytesSize());
        bMLU->copyFromHost(b.data(), indices->bytesSize());
        // Compute
        {
            void const *inputs[]{*aMLU, *bMLU};
            void *outputs[]{*cMLU};
            cnnlRoutine(res, *workspace, inputs, outputs);
            kernel::bang::sync();
        }
        {
            void const *inputs[]{a.data(), b.data()};
            void *outputs[]{c.data()};
            cpuRoutine(res, nullptr, inputs, outputs);
        }
        // Compare
        std::vector<float> result(output->elementsSize());
        cMLU->copyToHost(result.data(), output->bytesSize());
        for (auto i : range0_(c.size())) {
            EXPECT_FLOAT_EQ(c[i], result[i]);
        }
    }
}

#endif
