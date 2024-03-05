#ifdef USE_BANG

#include "../../../src/kernels/pad/cpu_kernel.hh"
#include "../../../src/kernels/pad/cnnl_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, PadCnnl) {
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
        auto kCpu = PadCpu::build(PadInfo(dims, *t1Tensor), type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        auto kernel = PadCnnl::build(dims, DataType::F32, type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        ASSERT_TRUE(kernel && kCpu);
        auto res = runtime::Resources();
        auto routine = kernel->lower(res).routine,
             rCpu = kCpu->lower(res).routine;
        // malloc
        auto &dev = *device::init(Device::Type::Mlu, 0, "");
        auto mluIn = dev.malloc(t1Tensor->bytesSize()),
             mluIn2 = dev.malloc(t2Tensor->bytesSize()),
             mluIn3 = dev.malloc(t3Tensor->bytesSize()),
             mluOut = dev.malloc(yTensor->bytesSize());
        // put input data
        std::vector<float> data(t1Tensor->elementsSize()),
            constvalue(1, 1.2f),
            cpuOut(yTensor->elementsSize());
        std::vector<int64_t> pads{1, 1, 0, 2, 1, 1, 0, 2};


        for (auto i : range0_(data.size())) { data[i] = i; }
        mluIn->copyFromHost(data.data(), t1Tensor->bytesSize());
        mluIn2->copyFromHost(pads.data(), t2Tensor->bytesSize());
        mluIn3->copyFromHost(constvalue.data(), t3Tensor->bytesSize());

        // inference
        {
            void const *inputs[]{*mluIn, *mluIn2, *mluIn3};
            void *outputs[]{*mluOut};
            routine(res, nullptr, inputs, outputs);
            kernel::bang::sync();
        }
        {
            void const *inputs[]{data.data(), pads.data(), constvalue.data()};
            void *outputs[]{cpuOut.data()};
            rCpu(res, nullptr, inputs, outputs);
        }
        // take output data
        std::vector<float> result(yTensor->elementsSize());
        mluOut->copyToHost(result.data(), yTensor->bytesSize());
        // check
        for (auto i : range0_(cpuOut.size())) {
            EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
        }
    }

    {
        PadDimension dims{
            {2, 2, 0},
            {3, 3, 0},
            {1, 1, 0},
            {4, 4, 0},
        };
        // build routine
        auto t1Tensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 4});
        auto t2Tensor = Tensor::share(DataType::I64, Shape{8});
        auto t3Tensor = Tensor::share(DataType::F32, Shape{});
        auto yTensor = Tensor::share(DataType::F32, Shape{2, 3, 1, 4});
        PadType type = PadType::Constant;
        auto kCpu = PadCpu::build(PadInfo(dims, *t1Tensor), type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        auto kernel = PadCnnl::build(dims, DataType::F32, type, std::make_optional(std::reference_wrapper(*t3Tensor)));
        ASSERT_TRUE(kernel && kCpu);
        auto res = runtime::Resources();
        auto routine = kernel->lower(res).routine,
             rCpu = kCpu->lower(res).routine;
        // malloc
        auto &dev = *device::init(Device::Type::Mlu, 0, "");
        auto mluIn = dev.malloc(t1Tensor->bytesSize()),
             mluIn2 = dev.malloc(t2Tensor->bytesSize()),
             mluIn3 = dev.malloc(t3Tensor->bytesSize()),
             mluOut = dev.malloc(yTensor->bytesSize());
        // put input data
        std::vector<float> data(t1Tensor->elementsSize()),
            constvalue(1, 1.2f),
            cpuOut(yTensor->elementsSize());
        std::vector<int64_t> pads{0, 0, 0, 0, 0, 0, 0, 0};


        for (auto i : range0_(data.size())) { data[i] = i; }
        mluIn->copyFromHost(data.data(), t1Tensor->bytesSize());
        mluIn2->copyFromHost(pads.data(), t2Tensor->bytesSize());
        mluIn3->copyFromHost(constvalue.data(), t3Tensor->bytesSize());

        // inference
        {
            void const *inputs[]{*mluIn, *mluIn2, *mluIn3};
            void *outputs[]{*mluOut};
            routine(res, nullptr, inputs, outputs);
            kernel::bang::sync();
        }
        {
            void const *inputs[]{data.data(), pads.data(), constvalue.data()};
            void *outputs[]{cpuOut.data()};
            rCpu(res, nullptr, inputs, outputs);
        }
        // take output data
        std::vector<float> result(yTensor->elementsSize());
        mluOut->copyToHost(result.data(), yTensor->bytesSize());
        // check
        for (auto i : range0_(cpuOut.size())) {
            EXPECT_FLOAT_EQ(cpuOut[i], result[i]);
        }
    }
}

#endif
