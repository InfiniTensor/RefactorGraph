#include "kernel/attributes/expand_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, ExpandInfo) {
    auto input = Tensor::share(DataType::F32, {3, 4, 1, 6}),
         output = Tensor::share(DataType::F32, {2, 3, 4, 5, 6});

    ExpandInfo info(*input, *output);
    EXPECT_EQ(info.blockSize, 24);
    EXPECT_EQ(info.blockCount, 120);
    EXPECT_EQ(info.strides, (std::vector<ExpandInfo::Dim>{{0, 60}, {1, 5}, {0, 1}}));

    auto reformed = info.reform(16);
    EXPECT_EQ(reformed.blockSize, 8);
    EXPECT_EQ(reformed.blockCount, 360);
    EXPECT_EQ(reformed.strides, (std::vector<ExpandInfo::Dim>{{0, 180}, {3, 15}, {0, 3}, {3, 3}, {1, 1}}));
}
