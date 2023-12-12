#include "kernel/attributes/scatter_nd_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, ScatterNDInfo) {
    auto data = Tensor::share(DataType::F32, {2, 3, 5, 7});
    auto indices = Tensor::share(DataType::I64, {3, 5, 2});
    ScatterNDInfo info(*data, *indices);
    EXPECT_EQ(info.prefix, 3 * 5);
    EXPECT_EQ(info.strides, (decltype(info.strides){3, 1}));
    EXPECT_EQ(info.blockSize, data->dataType.size() * 5 * 7);
}
