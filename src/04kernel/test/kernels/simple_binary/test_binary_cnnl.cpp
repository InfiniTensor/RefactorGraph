#ifdef USE_BANG

#include "../src/kernels/simple_binary/binary_cnnl.hh"
#include "../src/kernels/simple_binary/cpu_kernel.hh"
#include "../src/utilities/bang/cnrt_functions.h"
#include "hardware/device_manager.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;
using namespace hardware;

template<decltype(DataType::internal) T>
void testBinaryCnnl(SimpleBinaryType binaryOPT, Shape dimA, Shape dimB, Shape dimC) {
    // Create Tensor and build kernels
    using T_ = primitive<T>::type;
    auto aTensor = Tensor::share(T, dimA, LayoutType::NCHW);
    auto bTensor = Tensor::share(T, dimB, LayoutType::NCHW);
    auto cTensor = Tensor::share(T, dimC, LayoutType::NCHW);
    auto kernel = BinaryCnnl::build(binaryOPT, *aTensor, *bTensor, *cTensor);
    auto kCpu = BinaryCpu::build(binaryOPT, *aTensor, *bTensor);
    ASSERT_TRUE(kCpu && kernel);
    auto res = runtime::Resources();
    auto [routine, workspaceSize] = kernel->lower(res);
    auto rCpu = kCpu->lower(res).routine;
    // Init inputs and outputs
    std::vector<T_> a(aTensor->elementsSize(), 3);
    std::vector<T_> b(bTensor->elementsSize(), 2);
    std::vector<T_> c(cTensor->elementsSize());
    auto &dev = *device::init(Device::Type::Mlu, 0, "");
    auto workspace = dev.malloc(workspaceSize),
         aMLU = dev.malloc(aTensor->bytesSize()),
         bMLU = dev.malloc(bTensor->bytesSize()),
         cMLU = dev.malloc(cTensor->bytesSize());
    aMLU->copyFromHost(a.data(), aTensor->bytesSize());
    bMLU->copyFromHost(b.data(), bTensor->bytesSize());
    // Compute
    {
        void const *inputs[]{*aMLU, *bMLU};
        void *outputs[]{*cMLU};
        routine(res, *workspace, inputs, outputs);
        kernel::bang::sync();
    }
    {
        void const *inputs[]{a.data(), b.data()};
        void *outputs[]{c.data()};
        rCpu(res, nullptr, inputs, outputs);
    }
    // Compare
    std::vector<T_> result(cTensor->elementsSize());
    cMLU->copyToHost(result.data(), cTensor->bytesSize());
    for (auto i : range0_(result.size())) {
        EXPECT_EQ(c[i], result[i]);
    }
}
TEST(kernel, BinaryCnnlAdd) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Add,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlMul) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Mul,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlSub) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Sub,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlDiv) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Div,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlPow) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Pow,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlMod) {
    testBinaryCnnl<DataType::I32>(SimpleBinaryType::Mod,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlFMod) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Fmod,
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4},
                                  Shape{2, 5, 10, 20, 3, 4});
}

TEST(kernel, BinaryCnnlBroadcast) {
    testBinaryCnnl<DataType::F32>(SimpleBinaryType::Add, Shape{1, 2, 3, 4, 5, 6}, Shape{}, Shape{1, 2, 3, 4, 5, 6});
}


#endif
