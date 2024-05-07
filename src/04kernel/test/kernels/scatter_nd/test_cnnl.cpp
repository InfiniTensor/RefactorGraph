#ifdef USE_BANG

#include "../../../src/kernels/scatter_nd/cnnl_kernel.hh"
#include "../../../src/kernels/scatter_nd/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

TEST(kernel, ScatterNDCnnl) {
    // build routine
    auto data = Tensor::share(DataType::F32, Shape{8});
    auto indices = Tensor::share(DataType::I64, Shape{4, 1});
    auto updates = Tensor::share(DataType::F32, Shape{4});
    auto output = Tensor::share(DataType::F32, Shape{8});
    ScatterNDInfo info(*data, *indices);
    auto getRefs = [](std::vector<Arc<Tensor>> tensors) -> TensorRefs {
        TensorRefs refs;
        std::transform(tensors.begin(), tensors.end(), std::back_inserter(refs),
                       [](auto const &it) { return std::cref(*it); });
        return refs;
    };
    auto kernel = ScatterNDCnnl::build(getRefs({data, indices, updates}), getRefs({output})),
         kCpu = ScatterNDCpu::build(info);
    ASSERT_TRUE(kernel && kCpu);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine,
         rCpu = kCpu->lower(res).routine;
    // malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto mluData = dev.malloc(data->bytesSize()),
         mluIndices = dev.malloc(indices->bytesSize()),
         mluUpdates = dev.malloc(updates->bytesSize()),
         mluOut = dev.malloc(output->bytesSize());
    // put input data
    std::vector<float> data_(data->elementsSize());
    std::iota(data_.begin(), data_.end(), 1);
    std::vector<int64_t> indices_{4, 3, 1, 7};
    std::vector<float> updates_{9, 10, 11, 12};
    mluData->copyFromHost(data_.data(), data->bytesSize());
    mluIndices->copyFromHost(indices_.data(), indices->bytesSize());
    mluUpdates->copyFromHost(updates_.data(), updates->bytesSize());
    // inference
    {
        void const *inputs[]{*mluData, *mluIndices, *mluUpdates};
        void *outputs[]{*mluOut};
        routine(res, nullptr, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{data_.data(), indices_.data(), updates_.data()};
        void *outputs[]{data_.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // check
    std::vector<float> result(output->elementsSize());
    mluOut->copyToHost(result.data(), output->bytesSize());
    EXPECT_EQ(result, data_);
}

#endif
