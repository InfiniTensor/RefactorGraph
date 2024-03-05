#ifdef USE_BANG

#include "../../../src/kernels/split/cnnl_kernel.hh"
#include "../../../src/kernels/split/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, SplitCnnl) {
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
    auto info = SplitInfo(3, outputs_);
    auto kCpu = SplitCpu::build(info);
    auto kernel = SplitCnnl::build(3, *dataTensor, outputs_);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto rCpu = kCpu->lower(res).routine;
    auto [routine, workspaceSize]= kernel->lower(res);
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    Arc<Device::Blob>
        workspace = dev.malloc(workspaceSize),
        mluIn = dev.malloc(dataTensor->bytesSize()),
        mluOuts[]{
            dev.malloc(outputTensors[0]->bytesSize()),
            dev.malloc(outputTensors[1]->bytesSize()),
            dev.malloc(outputTensors[2]->bytesSize()),
            dev.malloc(outputTensors[3]->bytesSize()),
        };
    // put input data
    std::vector<float>
        data(dataTensor->elementsSize()),
        outsCpu[]{
            std::vector<float>(outputTensors[0]->elementsSize()),
            std::vector<float>(outputTensors[1]->elementsSize()),
            std::vector<float>(outputTensors[2]->elementsSize()),
            std::vector<float>(outputTensors[3]->elementsSize()),
        },
        outs[]{
            std::vector<float>(outputTensors[0]->elementsSize()),
            std::vector<float>(outputTensors[1]->elementsSize()),
            std::vector<float>(outputTensors[2]->elementsSize()),
            std::vector<float>(outputTensors[3]->elementsSize()),
        };
    std::iota(data.begin(), data.end(), 0);
    mluIn->copyFromHost(data.data(), dataTensor->bytesSize());
    // inference
    {
        void const *inputs[]{*mluIn};
        void *outputs[]{*mluOuts[0], *mluOuts[1], *mluOuts[2], *mluOuts[3]};
        routine(res, *workspace, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{data.data()};
        void *outputs[]{outsCpu[0].data(), outsCpu[1].data(), outsCpu[2].data(), outsCpu[3].data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    for (auto i : range0_(outputTensors.size())) {
        mluOuts[i]->copyToHost(outs[i].data(), outputTensors[i]->bytesSize());
        EXPECT_EQ(outs[i], outsCpu[i]);
    }
}

#endif
