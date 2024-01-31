#include "kernel/attributes/transpose_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, TransposeInfo) {
    {
        TransposeInfo info(
            DataType::F32,
            {1, 2, 3, 2, 1},
            {1, 2, 3, 0, 4});
        EXPECT_EQ(info.blockSize, 48);
        EXPECT_EQ(info.blockCount, 1);
        EXPECT_EQ(info.dims.size(), 1);
    }
    {
        TransposeInfo info(
            DataType::F32,
            {1, 1, 2, 1, 1},
            {1, 2, 3, 0, 4});
        EXPECT_EQ(info.blockSize, 8);
        EXPECT_EQ(info.blockCount, 1);
        EXPECT_EQ(info.dims.size(), 1);
    }
    {
        TransposeInfo info(
            DataType::F32,
            {1, 2, 3, 4, 5},
            {2, 3, 1, 0, 4});
        EXPECT_EQ(info.blockSize, 20);
        EXPECT_EQ(info.blockCount, 24);
        EXPECT_EQ(info.dims.size(), 2);
        EXPECT_EQ(info.dims[1].strideI, 12);
        EXPECT_EQ(info.dims[1].strideO, 1);
        EXPECT_EQ(info.dims[0].strideI, 1);
        EXPECT_EQ(info.dims[0].strideO, 2);
    }
}
