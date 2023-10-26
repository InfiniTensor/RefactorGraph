#include "kernel/attributes/gather_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, GatherInfo) {
    auto data = Tensor::share(DataType::F32, {2, 3, 10, 7, 8});
    auto indices = Tensor::share(DataType::I64, {4, 5, 6});
    GatherInfo info(2, *data, *indices);
    EXPECT_EQ(info.prefix, 2 * 3);
    EXPECT_EQ(info.postfix, 7 * 8 * DataType(DataType::F32).size());
    EXPECT_EQ(info.midSizeI, 10);
    EXPECT_EQ(info.midSizeO, 4 * 5 * 6);
}
