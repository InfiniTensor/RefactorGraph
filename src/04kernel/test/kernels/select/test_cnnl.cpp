#ifdef USE_BANG

#include "../../../src/kernels/select/cnnl_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <functional>
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;
using namespace hardware;

static void testSelect(const SelectType selectType, const std::vector<Shape> &shapes, const Shape &outShape, const std::vector<std::vector<float>> &data,
                       const std::vector<float> expectData) {
    // build routine
    TensorRefs dataTensors;
    std::vector<Tensor> tensorsVec;
    for (size_t i = 0; i < shapes.size(); ++i) {
        tensorsVec.push_back(Tensor(DataType::F32, shapes[i], LayoutType::Others, nullptr));
    }
    for (size_t i = 0; i < shapes.size(); ++i) {
        dataTensors.push_back(std::cref(tensorsVec[i]));
    }
    auto result = Tensor::share(DataType::F32, outShape);
    auto kernel = SelectCnnl::build(selectType, dataTensors);
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    // cnnl malloc
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    Arc<Device::Blob>
        workspace = dev.malloc(workspaceSize),
        mluIns[]{
            dev.malloc(dataTensors[0].get().bytesSize()),
            dev.malloc(dataTensors[1].get().bytesSize()),
            dev.malloc(dataTensors[2].get().bytesSize()),
        },
        mluOut = dev.malloc(result->bytesSize());
    // put input data
    mluIns[0]->copyFromHost(data[0].data(), dataTensors[0].get().bytesSize());
    mluIns[1]->copyFromHost(data[1].data(), dataTensors[1].get().bytesSize());
    mluIns[2]->copyFromHost(data[2].data(), dataTensors[2].get().bytesSize());
    // inference
    {
        void const *inputs[]{*mluIns[0], *mluIns[1], *mluIns[2]};
        void *outputs[]{*mluOut};
        routine(res, *workspace, inputs, outputs);
        kernel::bang::sync();
    }
    // check
    std::vector<float> out(result->elementsSize());
    mluOut->copyToHost(out.data(), result->bytesSize());
    for (auto i : range0_(expectData.size())) {
        EXPECT_FLOAT_EQ(expectData[i], out[i]);
    }
}

TEST(kernel, SelectCnnl) {
    // no need broadcast
    testSelect(SelectType::Max,
               {{1, 3}, {1, 3}, {1, 3}},
               {1, 3},
               {{3, 2, 1}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{1, 3}, {1, 3}, {1, 3}},
               {1, 3},
               {{3, 2, 1}, {1, 4, 4}, {2, 5, 3}},
               {1, 2, 1});

    // need broadcast
    testSelect(SelectType::Max,
               {{3}, {1, 3}, {1, 3}},
               {1, 3},
               {{3, 3, 3}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{3}, {1, 3}, {1, 3}},
               {1, 3},
               {{3, 3, 3}, {1, 4, 4}, {2, 5, 3}},
               {1, 3, 3});

    testSelect(SelectType::Max,
               {{1}, {1, 3}, {1, 3}},
               {1, 3},
               {{3}, {1, 4, 4}, {2, 5, 3}},
               {3, 5, 4});

    testSelect(SelectType::Min,
               {{1}, {1, 3}, {1, 3}},
               {1, 3},
               {{3}, {1, 4, 4}, {2, 5, 3}},
               {1, 3, 3});
}

#endif
